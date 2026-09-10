from __future__ import annotations
from dataclasses import dataclass
import numpy as np

@dataclass(frozen=True, slots=True)
class MINLPConformanceCheck:
    name: str
    passed: bool
    detail: str

@dataclass(frozen=True, slots=True)
class MINLPConformanceReport:
    schema: str
    checks: tuple[MINLPConformanceCheck, ...]
    passed: bool


def conform_minlp_orchestrator() -> MINLPConformanceReport:
    import solverpilot as om
    from .orchestrator import solve_binary_enumeration, solve_outer_approximation
    checks=[]
    m=om.Model('p8-conformance')
    x=m.variable(lower=-2,upper=2); z=m.binary()
    m.minimize((x-0.2)**2 + (z-0.4)**2)
    p=m.compile().execution_ir
    enum=solve_binary_enumeration(p); oa=solve_outer_approximation(p)
    checks.append(MINLPConformanceCheck('enumeration_global_proof',bool(enum.globally_proven),str(enum.status)))
    checks.append(MINLPConformanceCheck('oa_global_proof',bool(oa.globally_proven),str(oa.status)))
    agree=bool(enum.objective is not None and oa.objective is not None and abs(enum.objective-oa.objective)<=2e-6)
    checks.append(MINLPConformanceCheck('oa_matches_enumeration',agree,f'{enum.objective} vs {oa.objective}'))
    bound=bool(oa.lower_bound is not None and oa.objective is not None and oa.lower_bound <= oa.objective + 2e-6)
    checks.append(MINLPConformanceCheck('bound_consistency',bound,f'lb={oa.lower_bound}, ub={oa.objective}'))
    checks.append(MINLPConformanceCheck('no_global_claim_from_nlp_status_alone',all('global_subproblem_by_convexity_certificate' not in i.nlp_status for i in oa.iterations), 'global proof comes from convexity+bound closure'))
    return MINLPConformanceReport('solverpilot.minlp.conformance.v1',tuple(checks),all(c.passed for c in checks))
