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
from ._certificate_arithmetic import box_min, dot, matvec, rational, bounded_float


@dataclass(frozen=True, slots=True)
class OptimalityCheck:
    verified: bool
    primal_valid: bool
    dual_valid: bool
    stationarity_inf: float
    complementarity_inf: float
    gap: float
    reason: str
    dual_bound: float | None = None


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
        from solverpilot.problem.curvature import certify_psd
        curvature = certify_psd(problem.P)
        if not curvature.certified:
            return bad("independent PSD evidence unavailable: " + curvature.reason)
    bound = np.where(y > 0, hi, lo)
    active = y != 0
    if not np.isfinite(bound[active]).all():
        return bad('dual multiplier points toward an infinite bound')
    with np.errstate(over='ignore', invalid='ignore'):
        terms = np.zeros_like(y); terms[active] = y[active]*bound[active]
        stationarity = px + sign*linear.c + B.T@y
        comp = np.zeros_like(y); comp[active] = y[active]*(bound[active]-(B@x)[active])
        primal_value = float(.5*x@px + sign*linear.c@x)
        # A small stationarity residual can have a large effect on a wide box.
        # The tangent Lagrangian gives a valid lower bound after minimizing
        # this residual over the variable domain. Exact binary64 arithmetic
        # avoids losing a residual through cancellation in the sparse product.
        exact_px = matvec(problem.P, x) if isinstance(problem, QuadraticProblem) else [rational(0)] * len(x)
        exact_bty = matvec(B.T, y)
        residual = [p + rational(sign*c) + b for p, c, b in zip(exact_px, linear.c, exact_bty)]
        correction = box_min(residual, linear.variable_lower, linear.variable_upper)
        half_xpx = sum((rational(v)*p for v, p in zip(x, exact_px)), rational(0))/2
        exact_primal = half_xpx + dot(sign*linear.c, x)
        exact_dual = None if correction is None else -half_xpx-dot(y[active], bound[active])+correction
        if isinstance(problem, QuadraticProblem):
            # Gershgorin gives a conservative strong-convexity bound without
            # a dense eigendecomposition. It can bound free-variable residuals.
            matrix = problem.P.tocsr()
            lower_eigenvalue = min((sum((rational(matrix.data[k]) if matrix.indices[k] == i
                else -abs(rational(matrix.data[k])) for k in range(matrix.indptr[i], matrix.indptr[i+1])), rational(0))
                for i in range(len(x))), default=rational(0))
            if lower_eigenvalue > 0:
                strong_bound = (-half_xpx-dot(y[active], bound[active])
                    +sum((r*rational(v) for r, v in zip(residual, x)), rational(0))
                    -sum((r*r for r in residual), rational(0))/(2*lower_eigenvalue))
                exact_dual = strong_bound if exact_dual is None else max(exact_dual, strong_bound)
        dual_value = -np.inf if exact_dual is None else bounded_float(exact_dual)
        gap = np.inf if exact_dual is None else bounded_float(abs(exact_primal-exact_dual))
        st = float(np.max(np.abs(stationarity), initial=0))
        cp = float(np.max(np.abs(comp), initial=0))
        scale = np.maximum(1., np.abs(px)+np.abs(linear.c)+abs(B.T)@abs(y))
        st_ok = bool(np.all(np.abs(stationarity) <= tol.feasibility+tol.feasibility_rel*scale))
        allowed = tol.objective_abs+tol.objective_rel*max(1., abs(primal_value), abs(dual_value))
    finite = bool(np.isfinite([st, cp, gap, allowed]).all() and np.isfinite(scale).all())
    dual_ok = finite and st_ok
    verified = primal and dual_ok and gap <= allowed and cp <= allowed
    return OptimalityCheck(verified, primal, dual_ok, st, cp, gap,
                           'numerically verified KKT and domain-corrected gap' if verified else
                           ('stationarity error has no finite lower bound' if exact_dual is None else 'certificate outside tolerances'),
                           None if exact_dual is None else dual_value)


def verify_infeasibility(problem: LinearProblem, dual, *, atol=1e-8) -> bool:
    """Check a normalized Farkas separator for the continuous relaxation."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    B = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format='csr')
    lo, hi = np.r_[problem.constraint_lower, problem.variable_lower], np.r_[problem.constraint_upper, problem.variable_upper]
    try:
        y = np.asarray(dual, dtype=float)
    except (TypeError, ValueError):
        return False
    if y.shape != lo.shape or not np.isfinite(y).all() or not np.any(y):
        return False
    y = y / np.max(np.abs(y))
    active = y != 0; bound = np.where(y > 0, hi, lo)
    if not np.isfinite(bound[active]).all():
        return False
    residual = matvec(B.T, y)
    minimum = box_min(residual, problem.variable_lower, problem.variable_upper)
    separating = dot(y[active], bound[active])
    return minimum is not None and minimum-separating > rational(atol)


def verify_unboundedness(problem, point, direction, *, tolerances=None) -> bool:
    """Numerically check a feasible origin and descending recession direction."""
    tol = tolerances or ValidationTolerances()
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem) or linear.has_integer_variables:
        return False
    try:
        x, d = np.asarray(point, dtype=float), np.asarray(direction, dtype=float)
    except (TypeError, ValueError):
        return False
    if x.shape != (linear.n_variables,) or d.shape != x.shape or not np.isfinite(d).all() or not np.any(d):
        return False
    if not validate_solution(problem, CandidateSolution(x), tolerances=tol).valid:
        return False
    d = d / np.max(np.abs(d))
    B = sparse.vstack([linear.A, sparse.eye(linear.n_variables)], format='csr')
    lo, hi = np.r_[linear.constraint_lower, linear.variable_lower], np.r_[linear.constraint_upper, linear.variable_upper]
    # Recession signs concern an infinite ray: a tiny positive slope cannot
    # be treated as zero against a finite upper bound.
    for value, origin, lower, upper in zip(matvec(B, d), matvec(B, x), lo, hi):
        if np.isfinite(lower) and (value < 0 or origin < rational(lower)):
            return False
        if np.isfinite(upper) and (value > 0 or origin > rational(upper)):
            return False
    if isinstance(problem, QuadraticProblem):
        from solverpilot.problem.quadratic import require_confirmed_convexity
        require_confirmed_convexity(problem)
        if any(matvec(problem.P, d)):
            return False
    sign = 1. if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.
    return dot(sign*linear.c, d) < -rational(tol.objective_abs)
