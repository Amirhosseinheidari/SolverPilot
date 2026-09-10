from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from solverpilot.nlp.validation import NLPValidationReport, validate_nlp_solution
from .ir import MINLPProblem

@dataclass(frozen=True, slots=True)
class MINLPValidationReport:
    valid: bool
    nlp: NLPValidationReport
    max_integrality_violation: float
    binary_violations: tuple[int, ...]


def validate_minlp_solution(problem:MINLPProblem, x, *, objective_reported=None, atol=1e-7, rtol=1e-7):
    nlp=validate_nlp_solution(problem.relaxation,x,objective_reported=objective_reported,atol=atol,rtol=rtol)
    if x is None:
        return MINLPValidationReport(False,nlp,float('inf'),())
    a=np.asarray(x,dtype=float).reshape(-1)
    bad=[]; maxv=0.0
    for i in problem.binary_indices:
        v=float(a[i]); d=min(abs(v),abs(v-1.0)); maxv=max(maxv,d)
        if d>atol: bad.append(int(i))
    for i in problem.integer_indices:
        if i in problem.binary_indices: continue
        v=float(a[i]); d=abs(v-round(v)); maxv=max(maxv,d)
        if d>atol: bad.append(int(i))
    return MINLPValidationReport(bool(nlp.valid and not bad),nlp,float(maxv),tuple(sorted(set(bad))))
