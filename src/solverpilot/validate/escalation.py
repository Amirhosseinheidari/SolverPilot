"""Explicit certificate escalation under one caller-owned elapsed-time budget."""
from dataclasses import dataclass
import math
from time import perf_counter

from solverpilot.exact import ExactSolveResult, solve_exact
from .lp_dual import recover_lp_optimality
from .optimality import OptimalityCheck


@dataclass(frozen=True)
class LPRecoveryResult:
    optimality_verified: bool
    proof_kind: str
    numerical_check: OptimalityCheck
    exact_result: ExactSolveResult | None
    elapsed_s: float
    within_budget: bool


def recover_lp_with_fallback(problem, x, dual, *, basis=None, tolerances=None,
        time_limit_s=30., numerical_limit_s=10., scip_executable=None,
        checker_executable=None, evidence_directory=None):
    """Try numerical certification, then optional exact SCIP + independent VIPR.

    Exact fallback can return a different rational solution, held separately in
    exact_result. A verified bound alone never certifies an optimal solution.
    No native program is downloaded/discovered, and normal solve is unchanged.
    Numerical arithmetic has cooperative checks; a late result is not accepted.
    Use external process isolation when a hard return deadline is required.
    """
    for value in (time_limit_s, numerical_limit_s):
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError('recovery budgets must be finite and positive')
    if (scip_executable is None) != (checker_executable is None):
        raise ValueError('exact fallback requires both solver and independent checker')
    start = perf_counter()
    check = recover_lp_optimality(problem, x, dual, basis=basis, tolerances=tolerances,
                                 time_limit_s=min(numerical_limit_s, time_limit_s))
    exact = None
    remaining = time_limit_s-(perf_counter()-start)
    if not check.verified and remaining > 0 and scip_executable is not None:
        exact = solve_exact(problem, scip_executable=scip_executable,
            checker_executable=checker_executable, time_limit=remaining,
            evidence_directory=evidence_directory)
    elapsed = perf_counter()-start
    kind = ('numerical_tolerance' if check.verified else 'exact_optimal'
            if exact is not None and exact.independently_verified and exact.status == 'optimal'
            else 'verified_bound' if exact is not None and exact.independently_verified
            and exact.status == 'bound_verified' else 'unverified')
    within = elapsed <= time_limit_s
    return LPRecoveryResult(within and kind in {'numerical_tolerance', 'exact_optimal'},
                            kind, check, exact, elapsed, within)
