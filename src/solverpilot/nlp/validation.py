from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from .ad import NLPDerivativeEngine, evaluate_node
from .ir import NLPProblem

@dataclass(frozen=True, slots=True)
class NLPValidationReport:
    valid: bool
    max_bound_violation: float
    max_constraint_violation: float
    objective_recomputed: float | None
    objective_reported: float | None
    objective_difference: float | None
    objective_consistent: bool | None
    finite: bool
    warnings: tuple[str,...]=()


def _violation(v, lo, hi):
    return np.maximum(np.maximum(lo-v, v-hi), 0.0)

def validate_nlp_solution(problem:NLPProblem, x, *, objective_reported=None, atol=1e-7, rtol=1e-7):
    for name, raw in (("atol", atol), ("rtol", rtol)):
        if isinstance(raw, bool) or type(raw) not in (int, float) or not np.isfinite(float(raw)) or float(raw) < 0:
            raise ValueError(f'{name} must be finite and non-negative')
    atol=float(atol); rtol=float(rtol)
    if x is None:
        return NLPValidationReport(False, np.inf, np.inf, None, objective_reported, None, None, False, ('missing primal solution',))
    x=np.asarray(x,dtype=float).reshape(-1)
    if x.shape != (problem.n_variables,) or not np.all(np.isfinite(x)):
        return NLPValidationReport(False, np.inf, np.inf, None, objective_reported, None, None, False, ('non-finite or wrong-shaped primal solution',))
    bv=float(np.max(_violation(x,problem.variable_lower,problem.variable_upper))) if x.size else 0.0
    vals=[]
    finite=True
    for c in problem.constraints:
        val=np.asarray(evaluate_node(problem,c.node,x),dtype=float).reshape(-1)
        if not np.all(np.isfinite(val)): finite=False
        vals.append((val,c.lower,c.upper))
    cv=max((float(np.max(_violation(v,l,u))) for v,l,u in vals), default=0.0) if finite else np.inf
    obj_arr=np.asarray(evaluate_node(problem,problem.objective_node,x),dtype=float).reshape(-1)
    obj=float(obj_arr[0]) if obj_arr.size==1 and np.isfinite(obj_arr[0]) else None
    if obj is None:
        finite=False
    diff=None; consistent=None
    reported_finite=True
    if objective_reported is not None:
        try:
            reported_finite=np.isfinite(float(objective_reported))
        except (TypeError, ValueError):
            reported_finite=False
        if not reported_finite:
            finite=False
            diff=np.inf
            consistent=False
    if objective_reported is not None and obj is not None and reported_finite:
        diff=abs(float(objective_reported)-obj)
        consistent=diff <= atol + rtol*max(1.0,abs(obj),abs(float(objective_reported)))
    valid=finite and bv<=atol and cv<=atol and (consistent is not False)
    return NLPValidationReport(valid,bv,cv,obj,objective_reported,diff,consistent,finite,())
