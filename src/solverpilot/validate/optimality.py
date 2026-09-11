"""Numerical LP/convex-QP certificates in canonical original coordinates.

These are tolerance-qualified numerical checks, not exact-arithmetic proofs.
Dual ordering is linear rows then variable bounds; positive multipliers bind
upper bounds and negative multipliers bind lower bounds.
"""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from solverpilot.problem import LinearProblem, QuadraticProblem, ObjectiveSense
from .core import validate_solution
from .result import CandidateSolution, ValidationTolerances


@dataclass(frozen=True, slots=True)
class OptimalityCheck:
    verified: bool
    primal_valid: bool
    dual_valid: bool
    stationarity_inf: float
    complementarity_inf: float
    gap: float
    reason: str


def verify_optimality(problem, x, dual, *, tolerances=None) -> OptimalityCheck:
    tol = tolerances or ValidationTolerances()
    bad = lambda reason: OptimalityCheck(False, False, False, np.inf, np.inf, np.inf, reason)
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem) or linear.has_integer_variables:
        return bad('only continuous LP and convex QP are eligible')
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(dual, dtype=float)
    except (TypeError, ValueError):
        return bad("certificate is not numeric")
    B = sparse.vstack([linear.A, sparse.eye(linear.n_variables)], format='csr')
    lo, hi = np.r_[linear.constraint_lower, linear.variable_lower], np.r_[linear.constraint_upper, linear.variable_upper]
    if x.shape != (linear.n_variables,) or y.shape != lo.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return bad('certificate shape or finite-value check failed')
    primal = validate_solution(problem, CandidateSolution(x), tolerances=tol).valid
    sign = 1. if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.
    px = problem.P@x if isinstance(problem, QuadraticProblem) else np.zeros_like(x)
    if isinstance(problem, QuadraticProblem):
        from solverpilot.problem.quadratic import require_confirmed_convexity
        require_confirmed_convexity(problem)
    bound = np.where(y > 0, hi, lo)
    active = y != 0
    if not np.isfinite(bound[active]).all():
        return bad('dual multiplier points toward an infinite bound')
    with np.errstate(over='ignore', invalid='ignore'):
        terms = np.zeros_like(y); terms[active] = y[active]*bound[active]
        stationarity = px + sign*linear.c + B.T@y
        comp = np.zeros_like(y); comp[active] = y[active]*(bound[active]-(B@x)[active])
        primal_value = float(.5*x@px + sign*linear.c@x)
        dual_value = float(-.5*x@px - terms.sum())
        gap = abs(primal_value-dual_value)
        st = float(np.max(np.abs(stationarity), initial=0))
        cp = float(np.max(np.abs(comp), initial=0))
        scale = np.maximum(1., np.abs(px)+np.abs(linear.c)+abs(B.T)@abs(y))
        st_ok = bool(np.all(np.abs(stationarity) <= tol.feasibility+tol.feasibility_rel*scale))
        allowed = tol.objective_abs+tol.objective_rel*max(1., abs(primal_value), abs(dual_value))
    finite = bool(np.isfinite([st, cp, gap, allowed]).all() and np.isfinite(scale).all())
    dual_ok = finite and st_ok
    verified = primal and dual_ok and gap <= allowed and cp <= allowed
    return OptimalityCheck(verified, primal, dual_ok, st, cp, gap,
                           'numerically verified KKT and gap' if verified else 'certificate outside tolerances')


def verify_infeasibility(problem: LinearProblem, dual, *, atol=1e-8) -> bool:
    """Check a normalized Farkas separator for the continuous relaxation."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    B = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format='csr')
    lo, hi = np.r_[problem.constraint_lower, problem.variable_lower], np.r_[problem.constraint_upper, problem.variable_upper]
    y = np.asarray(dual, dtype=float)
    if y.shape != lo.shape or not np.isfinite(y).all() or not np.any(y):
        return False
    y = y / np.max(np.abs(y))
    active = y != 0; bound = np.where(y > 0, hi, lo)
    if not np.isfinite(bound[active]).all():
        return False
    residual, separating = B.T@y, float(y[active]@bound[active])
    return bool(np.isfinite(residual).all() and np.isfinite(separating)
                and np.max(np.abs(residual), initial=0) <= atol and separating < -atol)


def verify_unboundedness(problem, point, direction, *, tolerances=None) -> bool:
    """Numerically check a feasible origin and descending recession direction."""
    tol = tolerances or ValidationTolerances()
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem) or linear.has_integer_variables:
        return False
    x, d = np.asarray(point, dtype=float), np.asarray(direction, dtype=float)
    if x.shape != (linear.n_variables,) or d.shape != x.shape or not np.isfinite(d).all() or not np.any(d):
        return False
    if not validate_solution(problem, CandidateSolution(x), tolerances=tol).valid:
        return False
    d = d / np.max(np.abs(d))
    B = sparse.vstack([linear.A, sparse.eye(linear.n_variables)], format='csr')
    lo, hi = np.r_[linear.constraint_lower, linear.variable_lower], np.r_[linear.constraint_upper, linear.variable_upper]
    bd = B@d
    if not np.isfinite(bd).all():
        return False
    if np.any(bd[np.isfinite(lo)] < -tol.feasibility) or np.any(bd[np.isfinite(hi)] > tol.feasibility):
        return False
    if isinstance(problem, QuadraticProblem):
        from solverpilot.problem.quadratic import require_confirmed_convexity
        require_confirmed_convexity(problem)
        pd = problem.P@d
        if not np.isfinite(pd).all() or np.max(np.abs(pd), initial=0) > tol.feasibility:
            return False
    sign = 1. if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.
    descent = float(sign*linear.c@d)
    return bool(np.isfinite(descent) and descent < -tol.objective_abs)
