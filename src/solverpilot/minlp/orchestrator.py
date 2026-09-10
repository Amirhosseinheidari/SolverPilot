from __future__ import annotations
from dataclasses import dataclass
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



def _bound_closure(ub: float, lb: float, atol: float, rtol: float | None = None):
    rtol=float(atol if rtol is None else rtol)
    tol=float(atol)+rtol*max(abs(float(ub)),abs(float(lb)))
    primal_dual_gap=max(0.0,float(ub)-float(lb))
    bound_inversion=max(0.0,float(lb)-float(ub))
    return (max(primal_dual_gap,bound_inversion) <= tol, primal_dual_gap, bound_inversion, tol)

def _fixed_problem(base:NLPProblem, indices, values):
    lo=base.variable_lower.copy(); hi=base.variable_upper.copy()
    for i,v in zip(indices,values): lo[i]=hi[i]=float(v)
    return NLPProblem(base.n_variables,lo,hi,base.variable_layout,base.parameter_values,base.objective_node,base.constraints,base.objective_sense,base.name,{**dict(base.metadata),'fixed_binary_assignment':tuple(map(int,values))})


def solve_binary_enumeration(problem:MINLPProblem, *, nlp_backend=None, atol=1e-7, max_assignments=1<<18):
    if tuple(problem.integer_indices)!=tuple(problem.binary_indices): raise ValueError('binary enumeration supports binary discrete variables only')
    k=len(problem.binary_indices)
    if 2**k > max_assignments: raise ValueError('binary assignment count exceeds max_assignments')
    backend=nlp_backend or CasadiIpoptBackend()
    best=None; best_obj=np.inf; it=[]; unresolved=0
    for j,bits in enumerate(itertools.product((0,1), repeat=k),1):
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        r=backend.solve(sub)
        if r.local_optimal_candidate and r.validation.valid and r.x is not None:
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
    return MINLPSolveResult('proven_optimal' if proven else ('valid_feasible' if best is not None and valid else 'unknown'),best,None if best is None else best_obj,None,0.0 if proven else None,proven,valid,'binary_enumeration_convex_subproblems',tuple(it),{'assignments':2**k,'unresolved_assignments':unresolved,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions' if proven else 'none',False)


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
    return LinearProblem(A=A,c=cvec,variable_lower=vlm,variable_upper=vum,constraint_lower=np.asarray(cl),constraint_upper=np.asarray(cu),domains=domm,objective_sense=ObjectiveSense.MINIMIZE,objective_offset=offset,name='p8-oa-master'),has_nl_obj


def _oa_cuts_at(problem:MINLPProblem, x, *, include_objective=True):
    eng=NLPDerivativeEngine(problem.relaxation); f,g,grad,jac=eng.evaluate(x)
    cuts=[]; n=problem.n_variables; row_cursor=0
    for curvature,block in zip(problem.convexity_certificate.constraint_curvatures,problem.relaxation.constraints):
        size=block.lower.size
        if curvature=='affine': row_cursor += size; continue
        if size!=1: raise ValueError('P8 nonlinear OA cuts require scalar constraints')
        a=jac[row_cursor].copy(); gx=float(g[row_cursor]); intercept=gx-float(a@x)
        lo=float(block.lower[0]); hi=float(block.upper[0])
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


def solve_outer_approximation(problem:MINLPProblem, *, nlp_backend=None, milp_backend=None, max_iter=None, atol=1e-7):
    backend=nlp_backend or CasadiIpoptBackend(); mip=milp_backend or ScipyHighsBackend()
    iteration_budget = min((1 << len(problem.binary_indices)) + 5, 10000) if max_iter is None else int(max_iter)
    if iteration_budget <= 0: raise ValueError("max_iter must be positive")
    # Convex continuous relaxation supplies the initial global lower bound and first cuts.
    relax=backend.solve(problem.relaxation)
    if not (relax.local_optimal_candidate and relax.validation.valid and relax.x is not None):
        return MINLPSolveResult('unknown',None,None,None,None,False,False,'outer_approximation',(),{'reason':'continuous relaxation unresolved'})
    relax_lb=float(relax.validation.objective_recomputed)
    relax_minlp=validate_minlp_solution(problem,relax.x,objective_reported=relax_lb,atol=atol,rtol=atol)
    if relax_minlp.valid:
        return MINLPSolveResult('proven_optimal',relax.x.copy(),relax_lb,relax_lb,0.0,True,True,'outer_approximation',(),{'continuous_relaxation_lb':relax_lb,'integral_relaxation':True,'cuts':0,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions',False)
    # Ipopt may return an almost-integral relaxation point (e.g. 0.9999998).
    # Rounding alone is never a proof; fix the rounded assignment and re-solve.
    dist=max((min(abs(float(relax.x[i])),abs(float(relax.x[i])-1.0)) for i in problem.binary_indices),default=0.0)
    if dist <= max(1e-5,100*atol):
        bits=tuple(int(round(float(relax.x[i]))) for i in problem.binary_indices)
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        near=backend.solve(sub,x0=relax.x)
        if near.local_optimal_candidate and near.validation.valid and near.x is not None:
            ub0=float(near.validation.objective_recomputed)
            v0=validate_minlp_solution(problem,near.x,objective_reported=ub0,atol=atol,rtol=atol)
            if v0.valid and ub0-relax_lb <= atol*(1+abs(ub0)):
                return MINLPSolveResult('proven_optimal',near.x.copy(),ub0,relax_lb,max(0.0,ub0-relax_lb),True,True,'outer_approximation',(),{'continuous_relaxation_lb':relax_lb,'near_integral_relaxation_certified':True,'integrality_distance':dist,'cuts':0,'convexity_certificate':problem.convexity_certificate.schema},'solver_certified_under_convexity_assumptions',False)
    cuts=_oa_cuts_at(problem,relax.x)
    incumbent=None; ub=np.inf; lb=relax_lb; hist=[]; seen=set(); unresolved=False; evaluated_values=[]
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
        sub=_fixed_problem(problem.relaxation,problem.binary_indices,bits)
        nr=backend.solve(sub,x0=xmaster)
        added=0
        if nr.local_optimal_candidate and nr.validation.valid and nr.x is not None:
            obj=float(nr.validation.objective_recomputed)
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


def solve_minlp(problem:MINLPProblem, *, algorithm="oa", **kwargs):
    key=str(algorithm).lower().replace("-","_")
    if key in {"oa","outer_approximation"}:
        return solve_outer_approximation(problem, **kwargs)
    if key in {"enumeration","binary_enumeration"}:
        return solve_binary_enumeration(problem, **kwargs)
    raise ValueError(f"unsupported P8 MINLP algorithm {algorithm!r}")
