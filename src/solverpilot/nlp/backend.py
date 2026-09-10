from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import importlib.util
import numpy as np
from .ad import NLPDerivativeEngine, evaluate_node
from .ir import NLPProblem
from .validation import NLPValidationReport, validate_nlp_solution

@dataclass(frozen=True, slots=True)
class NLPSolveResult:
    backend:str
    backend_status:str
    x:np.ndarray|None
    objective_reported:float|None
    validation:NLPValidationReport
    local_optimal_candidate:bool
    globally_proven:bool
    multipliers_g:np.ndarray|None
    multipliers_x:np.ndarray|None
    kkt_stationarity_inf:float|None
    raw_statistics:dict[str,Any]

@dataclass(slots=True)
class CasadiIpoptBackend:
    max_iter:int=3000
    tol:float=1e-9
    print_level:int=0
    linear_solver:str|None=None
    @property
    def name(self): return 'casadi-ipopt-nlp-bridge'
    @property
    def binding_version(self):
        try:
            import casadi as ca; return ca.__version__
        except Exception: return None
    def is_available(self):
        if importlib.util.find_spec('casadi') is None: return False
        try:
            import casadi as ca; return bool(ca.has_nlpsol('ipopt'))
        except Exception: return False
    @property
    def capability_manifest_v2(self):
        from solverpilot import __version__
        from solverpilot.capabilities.v2 import BackendCapabilityManifestV2,CapabilityClaim,CapabilityEvidence,CapabilityKey,CapabilityMode,CapabilityStatus,VerificationLevel
        bv=f'bundled-with-casadi-{self.binding_version or "unknown"}'
        ev=CapabilityEvidence('p7-casadi-ipopt-conformance-v1','runtime-conformance',(bv,),(() if self.binding_version is None else (self.binding_version,)),(__version__,),'2026-09-06',('Ipopt local NLP solve + SolverPilot independent validation + derivative cross-check',))
        def v(restr=None):
            restr={} if restr is None else restr
            return CapabilityClaim(CapabilityStatus.RESTRICTED if restr else CapabilityStatus.SUPPORTED,CapabilityMode.EMULATED_SAFE,VerificationLevel.VERIFIED,restr,(ev,))
        claims={CapabilityKey.PROBLEM_NLP:v({'continuous_only':True,'local_solver':True,'global_optimality_not_claimed':True}),CapabilityKey.CONSTRAINT_NONLINEAR:v(),CapabilityKey.DERIVATIVE_GRADIENT:v(),CapabilityKey.DERIVATIVE_JACOBIAN:v(),CapabilityKey.DERIVATIVE_HESSIAN:v(),CapabilityKey.RESULT_PRIMAL:v(),CapabilityKey.RESULT_OBJECTIVE:v(),CapabilityKey.RESULT_DUAL:v({'backend_multiplier_sign_convention':'Ipopt/CasADi'}),CapabilityKey.START_PRIMAL:v()}
        return BackendCapabilityManifestV2(self.name,bv,'casadi',self.binding_version,__version__,claims=claims,metadata={'plugin':'ipopt','local_nlp':True})
    def solve(self,problem:NLPProblem,*,x0=None):
        if not self.is_available(): raise RuntimeError('CasADi Ipopt plugin unavailable')
        import casadi as ca
        eng=NLPDerivativeEngine(problem)
        nlp={'x':eng.x,'f':eng.f}
        if problem.n_constraints: nlp['g']=eng.g
        opts={'print_time':False,'ipopt.print_level':int(self.print_level),'ipopt.max_iter':int(self.max_iter),'ipopt.tol':float(self.tol)}
        if self.linear_solver: opts['ipopt.linear_solver']=self.linear_solver
        solver=ca.nlpsol('solverpilot_p7_ipopt','ipopt',nlp,opts)
        initial_search_attempts=0
        if x0 is None:
            lo,hi=problem.variable_lower,problem.variable_upper
            midpoint=np.where(np.isfinite(lo)&np.isfinite(hi),0.5*(lo+hi),np.where(np.isfinite(lo),np.maximum(lo,0),np.where(np.isfinite(hi),np.minimum(hi,0),0.0)))
            candidates=[midpoint]
            # Deterministic interior-biased alternatives. They are only a starting-point
            # strategy; final validity still comes from independent original-space validation.
            for frac in (0.25,0.75):
                cand=np.where(np.isfinite(lo)&np.isfinite(hi),lo+frac*(hi-lo),midpoint)
                candidates.append(cand)
            candidates.append(np.clip(np.ones(problem.n_variables),lo,hi))
            candidates.append(np.clip(-np.ones(problem.n_variables),lo,hi))
            x0=midpoint
            for cand in candidates:
                initial_search_attempts += 1
                try:
                    f=evaluate_node(problem,problem.objective_node,cand)
                    gs=[evaluate_node(problem,b.node,cand) for b in problem.constraints]
                    if np.isfinite(np.asarray(f,dtype=float)).all() and all(np.isfinite(np.asarray(g,dtype=float)).all() for g in gs):
                        x0=np.asarray(cand,dtype=float); break
                except (FloatingPointError,ValueError,TypeError,OverflowError):
                    continue
        kw={'x0':np.asarray(x0,float),'lbx':problem.variable_lower,'ubx':problem.variable_upper}
        if problem.n_constraints: kw.update(lbg=problem.constraint_lower,ubg=problem.constraint_upper)
        result=None; err=None
        try: result=solver(**kw)
        except Exception as e: err=f'{type(e).__name__}: {e}'
        try: stats=dict(solver.stats())
        except Exception: stats={}
        x=None; obj=None; lg=None; lx=None
        if result is not None:
            if 'x' in result:
                arr=np.asarray(result['x'],dtype=float).reshape(-1)
                if arr.shape==(problem.n_variables,) and np.all(np.isfinite(arr)): x=arr
            if 'f' in result:
                try: obj=float(result['f'])
                except Exception: pass
            if 'lam_g' in result: lg=np.asarray(result['lam_g'],dtype=float).reshape(-1)
            if 'lam_x' in result: lx=np.asarray(result['lam_x'],dtype=float).reshape(-1)
        validation=validate_nlp_solution(problem,x,objective_reported=obj,atol=max(1e-7,self.tol*100),rtol=max(1e-7,self.tol*100))
        raw=str(stats.get('return_status',''))
        stationarity=None
        if x is not None and lg is not None and lx is not None:
            try:
                grad=eng.gradient(x); jac=eng.jacobian(x)
                stationarity=float(np.max(np.abs(grad + jac.T @ lg + lx)))
            except Exception:
                stationarity=None
        kkt_tol=max(1e-6,self.tol*1000)
        success=bool(stats.get('success')) and validation.valid and stationarity is not None and stationarity <= kkt_tol
        stats.update({'error':err,'initial_point_domain_search_attempts':initial_search_attempts,'domain_hazards':list(problem.metadata.get('domain_hazards',())),'casadi_version':self.binding_version,'nlpsol_plugin':'ipopt','global_optimality_claimed':False,'kkt_stationarity_inf':stationarity,'kkt_stationarity_tolerance':kkt_tol})
        return NLPSolveResult(self.name,raw,x,obj,validation,success,False,lg,lx,stationarity,stats)
