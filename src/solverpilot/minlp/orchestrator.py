from __future__ import annotations
from dataclasses import dataclass, field
from solverpilot._identity import new_execution_id
from typing import Any
import itertools
import numpy as np
from scipy import sparse
from solverpilot.backends import ScipyHighsBackend
from solverpilot.runtime.executor import execute
from solverpilot.problem import LinearProblem, ObjectiveSense, VariableDomain
from solverpilot.nlp import CasadiIpoptBackend, NLPDerivativeEngine, validate_nlp_solution
from solverpilot.nlp.ir import NLPProblem
from solverpilot.validate import PublicStatus
from .ir import MINLPProblem
from .validation import validate_minlp_solution

@dataclass(frozen=True, slots=True)
class MINLPIteration:
    iteration:int
    binary_assignment:tuple[int,...]
    master_lower_bound:float|None
    nlp_objective:float|None
    nlp_status:str
    cuts_added:int

@dataclass(frozen=True, slots=True)
class MINLPSolveResult:
    status:str
    x:np.ndarray|None
    objective:float|None
    lower_bound:float|None
    gap:float|None
    globally_proven:bool
    validation_valid:bool
    algorithm:str
    iterations:tuple[MINLPIteration,...]
    raw_statistics:dict[str,Any]
    proof_scope:str='none'
    independently_verified_global:bool=False
    problem_data_hash: str | None = None
    execution_id: str = field(default_factory=new_execution_id, kw_only=True, compare=False)

    def __post_init__(self):
        from solverpilot._immutability import deep_freeze, readonly_array
        if self.x is not None:
            object.__setattr__(self, 'x', readonly_array(self.x, dtype=float))
        object.__setattr__(self, 'raw_statistics', deep_freeze(self.raw_statistics))



def _bound_closure(ub: float, lb: float, atol: float, rtol: float | None = None):
    rtol=float(atol if rtol is None else rtol)
    if not all(np.isfinite(v) for v in (ub, lb, atol, rtol)) or min(atol, rtol) < 0:
        return False, float('inf'), float('inf'), 0.0
    tol=float(atol)+rtol*max(abs(float(ub)),abs(float(lb)))
    if not np.isfinite(tol):
        return False, float('inf'), float('inf'), 0.0
    primal_dual_gap=max(0.0,float(ub)-float(lb))
    bound_inversion=max(0.0,float(lb)-float(ub))
    return (max(primal_dual_gap,bound_inversion) <= tol, primal_dual_gap, bound_inversion, tol)

def _fixed_problem(base:NLPProblem, indices, values):
    lo=base.variable_lower.copy(); hi=base.variable_upper.copy()
    if len(indices) != len(values): raise ValueError('assignment length mismatch')
    for i,v in zip(indices,values):
        if not lo[i] <= v <= hi[i]: raise ValueError('fixed assignment violates original variable bounds')
        lo[i]=hi[i]=float(v)
    return NLPProblem(base.n_variables,lo,hi,base.variable_layout,base.parameter_values,base.objective_node,base.constraints,base.objective_sense,base.name,{**dict(base.metadata),'fixed_binary_assignment':tuple(map(int,values))})


def _solve_binary_enumeration(problem:MINLPProblem, *, nlp_backend=None, atol=1e-7, max_assignments=1<<18, _progress_state=None):
    if tuple(problem.integer_indices)!=tuple(problem.binary_indices): raise ValueError('binary enumeration supports binary discrete variables only')
    domains = tuple(tuple(v for v in (0,1) if problem.relaxation.variable_lower[i] <= v <= problem.relaxation.variable_upper[i]) for i in problem.binary_indices)
    count = int(np.prod([len(d) for d in domains], dtype=object))
    if count > max_assignments: raise ValueError('binary assignment count exceeds max_assignments')
    backend=nlp_backend or CasadiIpoptBackend()
    best=None; best_obj=np.inf; it=[]; unresolved=0
    if _progress_state is not None: _progress_state['iterations']=it; _progress_state['assignment_count']=count
    for j,bits in enumerate(itertools.product(*domains),1):
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        r=backend.solve(sub)
        if r.local_optimal_candidate and r.validation.valid and r.x is not None and validate_minlp_solution(problem,r.x,objective_reported=r.validation.objective_recomputed,atol=atol,rtol=atol).valid:
            obj=float(r.validation.objective_recomputed)
            if obj < best_obj-atol: best_obj=obj; best=r.x.copy()
            st='global_subproblem_by_convexity_certificate'
        else:
            # Without an NLP infeasibility certificate a failed local solve cannot
            # safely eliminate an assignment from a global proof.
            unresolved += 1; st='unresolved'
        it.append(MINLPIteration(j,tuple(map(int,bits)),None,None if r.x is None else r.validation.objective_recomputed,r.backend_status,0))
    valid=False
    if best is not None: valid=validate_minlp_solution(problem,best,objective_reported=best_obj,atol=atol,rtol=atol).valid
    proven=best is not None and unresolved==0 and valid
    return MINLPSolveResult('proven_optimal' if proven else ('valid_feasible' if best is not None and valid else 'unknown'),best,None if best is None else best_obj,None,0.0 if proven else None,proven,valid,'binary_enumeration_convex_subproblems',tuple(it),{'assignments':count,'unresolved_assignments':unresolved,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions' if proven else 'none',False)


def _build_master(problem:MINLPProblem, cuts, eta_lower, reference_x):
    base=problem.relaxation; n=problem.n_variables
    eng=NLPDerivativeEngine(base); f,g,grad,jac=eng.evaluate(reference_x)
    has_nl_obj=problem.convexity_certificate.objective_curvature != 'affine'
    nmaster=n+1 if has_nl_obj else n
    rows=[]; cl=[]; cu=[]
    row_cursor=0
    for curvature,block in zip(problem.convexity_certificate.constraint_curvatures,base.constraints):
        size=block.lower.size
        if curvature=='affine':
            for local in range(size):
                a=jac[row_cursor+local].copy(); val=float(g[row_cursor+local]); intercept=val-float(a@reference_x)
                rows.append({i:float(v) for i,v in enumerate(a) if v!=0})
                lo=float(block.lower[local]); hi=float(block.upper[local])
                cl.append(lo-intercept if np.isfinite(lo) else -np.inf)
                cu.append(hi-intercept if np.isfinite(hi) else np.inf)
        row_cursor += size
    for row,lo,hi in cuts: rows.append(dict(row)); cl.append(float(lo)); cu.append(float(hi))
    data=[]; rr=[]; cc=[]
    for r,row in enumerate(rows):
        for j,v in row.items():
            if v!=0: rr.append(r); cc.append(j); data.append(float(v))
    A=sparse.csr_matrix((data,(rr,cc)),shape=(len(rows),nmaster))
    cvec=np.zeros(nmaster)
    if has_nl_obj:
        cvec[n]=1.0; vlm=np.r_[base.variable_lower,float(eta_lower)]; vum=np.r_[base.variable_upper,np.inf]; domm=tuple(problem.domains)+('continuous',); offset=0.0
    else:
        cvec[:n]=grad; offset=float(f-grad@reference_x); vlm=base.variable_lower; vum=base.variable_upper; domm=tuple(problem.domains)
    vlm=vlm.copy(); vum=vum.copy()
    for i in problem.binary_indices:
        allowed=[v for v in (0,1) if base.variable_lower[i]<=v<=base.variable_upper[i]]
        if not allowed: raise ValueError('binary variable has no admissible value')
        vlm[i]=min(allowed); vum[i]=max(allowed)
    return LinearProblem(A=A,c=cvec,variable_lower=vlm,variable_upper=vum,constraint_lower=np.asarray(cl),constraint_upper=np.asarray(cu),domains=domm,objective_sense=ObjectiveSense.MINIMIZE,objective_offset=offset,name='p8-oa-master'),has_nl_obj


def _oa_cuts_at(problem:MINLPProblem, x, *, include_objective=True):
    eng=NLPDerivativeEngine(problem.relaxation); f,g,grad,jac=eng.evaluate(x)
    cuts=[]; n=problem.n_variables; row_cursor=0
    for curvature,block in zip(problem.convexity_certificate.constraint_curvatures,problem.relaxation.constraints):
        size=block.lower.size
        if curvature=='affine': row_cursor += size; continue
        for local in range(size):
            a=jac[row_cursor+local].copy(); gx=float(g[row_cursor+local]); intercept=gx-float(a@x)
            lo=float(block.lower[local]); hi=float(block.upper[local])
            if np.isfinite(hi) and not np.isfinite(lo): cuts.append(({i:float(v) for i,v in enumerate(a) if v!=0},-np.inf,hi-intercept))
            elif np.isfinite(lo) and not np.isfinite(hi): cuts.append(({i:float(v) for i,v in enumerate(a) if v!=0},lo-intercept,np.inf))
            else: raise ValueError('nonlinear equality/two-sided cut unsupported')
        row_cursor += size
    if include_objective and problem.convexity_certificate.objective_curvature != 'affine':
        row={i:float(-v) for i,v in enumerate(grad) if v!=0}; row[n]=1.0
        rhs=float(f-grad@x); cuts.append((row,rhs,np.inf))
    return cuts


def _binary_nogood_cut(problem:MINLPProblem, bits):
    row={}; ones=0
    for idx,bit in zip(problem.binary_indices,bits):
        if int(bit)==1:
            row[int(idx)]=-1.0; ones+=1
        else:
            row[int(idx)]=1.0
    return (row,1.0-float(ones),np.inf)


def _solve_outer_approximation(problem:MINLPProblem, *, nlp_backend=None, milp_backend=None, max_iter=None, atol=1e-7, _progress_state=None):
    if isinstance(atol, bool) or not isinstance(atol, (int, float)) or not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and non-negative')
    if max_iter is not None and (isinstance(max_iter, bool) or not isinstance(max_iter, int) or max_iter < 1):
        raise ValueError('max_iter must be a positive integer')
    domains=[tuple(v for v in (0,1) if problem.relaxation.variable_lower[i]<=v<=problem.relaxation.variable_upper[i]) for i in problem.binary_indices]
    if any(not d for d in domains):
        return MINLPSolveResult('unknown',None,None,None,None,False,False,'outer_approximation',(),{'reason':'binary variable has no admissible value'})
    if domains and all(len(d)==1 for d in domains):
        from dataclasses import replace
        return replace(_solve_binary_enumeration(problem,nlp_backend=nlp_backend,atol=atol,max_assignments=1,_progress_state=_progress_state),algorithm='outer_approximation')
    backend=nlp_backend or CasadiIpoptBackend(); mip=milp_backend or ScipyHighsBackend()
    iteration_budget = min((1 << len(problem.binary_indices)) + 5, 10000) if max_iter is None else int(max_iter)
    if iteration_budget <= 0: raise ValueError("max_iter must be positive")
    # Convex continuous relaxation supplies the initial global lower bound and first cuts.
    relax=backend.solve(problem.relaxation)
    if not (relax.local_optimal_candidate and relax.validation.valid and relax.x is not None):
        return MINLPSolveResult('unknown',None,None,None,None,False,False,'outer_approximation',(),{'reason':'continuous relaxation unresolved'})
    relax_lb=float(relax.validation.objective_recomputed)
    if not np.isfinite(relax_lb):
        return MINLPSolveResult('unknown',None,None,None,None,False,False,'outer_approximation',(),{'reason':'non-finite continuous relaxation bound'})
    relax_minlp=validate_minlp_solution(problem,relax.x,objective_reported=relax_lb,atol=atol,rtol=atol)
    if relax_minlp.valid and all(float(relax.x[i]) in (0.,1.) for i in problem.binary_indices):
        return MINLPSolveResult('proven_optimal',relax.x.copy(),relax_lb,relax_lb,0.0,True,True,'outer_approximation',(),{'continuous_relaxation_lb':relax_lb,'integral_relaxation':True,'cuts':0,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions',False)
    # Ipopt may return an almost-integral relaxation point (e.g. 0.9999998).
    # Rounding alone is never a proof; fix the rounded assignment and re-solve.
    dist=max((min(abs(float(relax.x[i])),abs(float(relax.x[i])-1.0)) for i in problem.binary_indices),default=0.0)
    rounded=tuple(int(round(float(relax.x[i]))) for i in problem.binary_indices)
    admissible=all(problem.relaxation.variable_lower[i]<=v<=problem.relaxation.variable_upper[i] for i,v in zip(problem.binary_indices,rounded))
    if dist <= max(1e-5,100*atol) and admissible:
        bits=tuple(int(round(float(relax.x[i]))) for i in problem.binary_indices)
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        near=backend.solve(sub,x0=relax.x)
        if near.local_optimal_candidate and near.validation.valid and near.x is not None:
            ub0=float(near.validation.objective_recomputed)
            v0=validate_minlp_solution(problem,near.x,objective_reported=ub0,atol=atol,rtol=atol)
            closed, gap, inversion, proof_tol = _bound_closure(ub0, relax_lb, atol)
            if v0.valid and inversion > proof_tol:
                # A feasible fixed assignment below the relaxation lower bound
                # disproves that bound. Do not feed it into an OA master.
                return MINLPSolveResult('valid_feasible',near.x.copy(),ub0,None,None,False,True,'outer_approximation',(),{'reason':'continuous relaxation bound contradicts feasible assignment','rejected_continuous_relaxation_lb':relax_lb,'bound_inversion':inversion,'proof_tolerance':proof_tol,'integrality_distance':dist})
            if v0.valid and closed:
                return MINLPSolveResult('proven_optimal',near.x.copy(),ub0,relax_lb,gap,True,True,'outer_approximation',(),{'continuous_relaxation_lb':relax_lb,'near_integral_relaxation_certified':True,'integrality_distance':dist,'bound_inversion':inversion,'proof_tolerance':proof_tol,'cuts':0,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions',False)
    cuts=_oa_cuts_at(problem,relax.x)
    incumbent=None; ub=np.inf; lb=relax_lb; hist=[]; seen=set(); unresolved=False; evaluated_values=[]
    if _progress_state is not None: _progress_state['iterations']=hist
    for itn in range(1,iteration_budget+1):
        master,has_eta=_build_master(problem,cuts,relax_lb,relax.x)
        mr=execute(master,mip)
        if mr.status is PublicStatus.INFEASIBLE:
            if incumbent is not None and evaluated_values:
                lb=ub
                valid=validate_minlp_solution(problem,incumbent,objective_reported=ub,atol=atol,rtol=atol).valid
                return MINLPSolveResult('proven_optimal' if valid else 'unknown',incumbent,ub,lb,0.0,valid,valid,'outer_approximation',tuple(hist),{'continuous_relaxation_lb':relax_lb,'cuts':len(cuts),'master_exhausted':True,'master_infeasibility_backend_reported':True,'master_infeasibility_certificate_verified':False,'evaluated_assignments':len(evaluated_values),'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions' if valid else 'none',False)
            unresolved=True; break
        if mr.status is not PublicStatus.VALID_OPTIMAL or mr.x is None or mr.objective is None:
            unresolved=True; break
        remaining_lb=max(relax_lb,float(mr.objective))
        lb=min(remaining_lb,min(evaluated_values)) if evaluated_values else remaining_lb
        xmaster=np.asarray(mr.x[:problem.n_variables],float)
        bits=tuple(int(round(xmaster[i])) for i in problem.binary_indices)
        if not all(problem.relaxation.variable_lower[i]<=v<=problem.relaxation.variable_upper[i] for i,v in zip(problem.binary_indices,bits)):
            unresolved=True; break
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        nr=backend.solve(sub,x0=xmaster)
        added=0
        if nr.local_optimal_candidate and nr.validation.valid and nr.x is not None:
            obj=float(nr.validation.objective_recomputed)
            valid_sub = validate_minlp_solution(problem,nr.x,objective_reported=obj,atol=atol,rtol=atol).valid
            _, _, inversion, proof_tol = _bound_closure(obj, relax_lb, atol)
            if valid_sub and inversion > proof_tol:
                return MINLPSolveResult('valid_feasible',nr.x.copy(),obj,None,None,False,True,'outer_approximation',tuple(hist),{'reason':'continuous relaxation bound contradicts feasible assignment','rejected_continuous_relaxation_lb':relax_lb,'bound_inversion':inversion,'proof_tolerance':proof_tol,'evaluated_assignments':len(evaluated_values)+1})
            if not valid_sub:
                unresolved=True
                break
            if obj < ub-atol: ub=obj; incumbent=nr.x.copy()
            evaluated_values.append(obj)
            newcuts=_oa_cuts_at(problem,nr.x)
            newcuts.append(_binary_nogood_cut(problem,bits))
            cuts.extend(newcuts); added=len(newcuts)
        else:
            # No infeasibility certificate: do not add a no-good cut and do not claim a proof.
            unresolved=True
        hist.append(MINLPIteration(itn,bits,lb,None if nr.x is None else nr.validation.objective_recomputed,nr.backend_status,added))
        if unresolved: break
        if np.isfinite(ub):
            closed,gap,inversion,proof_tol=_bound_closure(ub,lb,atol)
        else:
            closed,gap,inversion,proof_tol=False,np.inf,0.0,np.inf
        if np.isfinite(ub) and closed:
            valid=validate_minlp_solution(problem,incumbent,objective_reported=ub,atol=atol,rtol=atol).valid
            return MINLPSolveResult('proven_optimal' if valid else 'unknown',incumbent,ub,lb,gap,valid,valid,'outer_approximation',tuple(hist),{'continuous_relaxation_lb':relax_lb,'cuts':len(cuts),'evaluated_assignments':len(evaluated_values),'bound_inversion':inversion,'proof_tolerance':proof_tol,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions' if valid else 'none',False)
        key=(bits,round(lb,10),round(ub,10) if np.isfinite(ub) else None)
        if key in seen:
            unresolved=True; break
        seen.add(key)
    valid=False
    if incumbent is not None: valid=validate_minlp_solution(problem,incumbent,objective_reported=ub,atol=atol,rtol=atol).valid
    if not np.isfinite(ub): gap=None; inversion=0.0; proof_tol=None
    else:
        _,gap,inversion,proof_tol=_bound_closure(ub,lb,atol)
    return MINLPSolveResult('valid_feasible' if valid else 'unknown',incumbent,None if incumbent is None else ub,lb,gap,False,valid,'outer_approximation',tuple(hist),{'continuous_relaxation_lb':relax_lb,'cuts':len(cuts),'evaluated_assignments':len(evaluated_values),'unresolved':unresolved,'iteration_budget':iteration_budget,'bound_inversion':inversion,'proof_tolerance':proof_tol,'convexity_certificate':problem.convexity_certificate.schema})


class _AggregateExpired(TimeoutError):
    pass


def _aggregate_minlp(problem, function, algorithm, budget, kwargs):
    from dataclasses import replace
    from solverpilot.runtime.budgeting import CallBudget, configured_backend, apply_budget
    from solverpilot.exceptions import BudgetNotSupportedError
    tracker=CallBudget(budget)
    if budget is not None and (budget.memory_mb is not None or budget.threads is not None):
        raise BudgetNotSupportedError('MINLP aggregate memory/thread budgets are not implemented')
    best=None; best_obj=np.inf; atol=kwargs.get('atol',1e-7)
    state={'iterations':[]}; kwargs['_progress_state']=state

    class TimedBackend:
        def __init__(self, backend, nlp): self.backend=backend; self.nlp=nlp
        def __getattr__(self,name): return getattr(self.backend,name)
        def solve(self,sub,**controls):
            nonlocal best,best_obj
            try: remaining=tracker.remaining_budget()
            except TimeoutError as exc: raise _AggregateExpired(str(exc)) from exc
            if self.nlp:
                backend=self.backend
                if remaining is not None and remaining.wall_time_s is not None:
                    if not hasattr(backend,'time_limit_s'): raise BudgetNotSupportedError('NLP backend has no native time budget')
                    backend=configured_backend(backend,{'time_limit_s':remaining.wall_time_s})
            else: backend=apply_budget(self.backend,remaining)
            result=backend.solve(sub,**controls)
            if self.nlp and result.x is not None:
                validation=validate_minlp_solution(problem,result.x,atol=atol,rtol=atol)
                if validation.valid and validation.nlp.objective_recomputed < best_obj:
                    best_obj=float(validation.nlp.objective_recomputed); best=result.x.copy()
            return result

    if budget is not None:
        kwargs['nlp_backend']=TimedBackend(kwargs.get('nlp_backend') or CasadiIpoptBackend(),True)
        if algorithm=='outer_approximation': kwargs['milp_backend']=TimedBackend(kwargs.get('milp_backend') or ScipyHighsBackend(),False)
    try: result=function(problem,**kwargs)
    except _AggregateExpired:
        result=MINLPSolveResult('valid_feasible' if best is not None else 'unknown',best,None if best is None else best_obj,
                               None,None,False,best is not None,algorithm,tuple(state['iterations']),
                               {'reason':'aggregate budget exhausted','assignments':state.get('assignment_count'),
                                'completed_iterations':len(state['iterations']),'convexity_certificate':problem.convexity_certificate.schema})
    return replace(result,problem_data_hash=problem.data_hash,
                   raw_statistics={**dict(result.raw_statistics),'call_budget':tracker.report()})


def solve_binary_enumeration(problem:MINLPProblem, *, nlp_backend=None, atol=1e-7, max_assignments=1<<18, budget=None):
    return _aggregate_minlp(problem,_solve_binary_enumeration,'binary_enumeration_convex_subproblems',budget,
                           dict(nlp_backend=nlp_backend,atol=atol,max_assignments=max_assignments))


def solve_outer_approximation(problem:MINLPProblem, *, nlp_backend=None, milp_backend=None, max_iter=None, atol=1e-7, budget=None):
    return _aggregate_minlp(problem,_solve_outer_approximation,'outer_approximation',budget,
                           dict(nlp_backend=nlp_backend,milp_backend=milp_backend,max_iter=max_iter,atol=atol))


def solve_minlp(problem:MINLPProblem, *, algorithm="oa", **kwargs):
    key=str(algorithm).lower().replace("-","_")
    if key in {"oa","outer_approximation"}:
        return solve_outer_approximation(problem, **kwargs)
    if key in {"enumeration","binary_enumeration"}:
        return solve_binary_enumeration(problem, **kwargs)
    raise ValueError(f"unsupported P8 MINLP algorithm {algorithm!r}")
