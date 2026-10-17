"""Numerical LP/convex-QP certificates in canonical original coordinates.

These are tolerance-qualified numerical checks, not exact-arithmetic proofs.
Dual ordering is linear rows then variable bounds; positive multipliers bind
upper bounds and negative multipliers bind lower bounds.
"""
from dataclasses import dataclass
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem

from ._certificate_arithmetic import bounded_float, box_min, dot, matvec, rational
from .core import validate_solution
from .result import CandidateSolution, ValidationReport, ValidationTolerances
from .workspace import controlled_verification


@dataclass(frozen=True, slots=True)
class OptimalityCheck:
    """Numerical evidence with a bound in the original objective's coordinates.

    ``dual_bound`` includes the objective offset and is a lower bound for
    minimization or an upper bound for maximization. Its binary64 representation
    is rounded outward; an outward infinity can represent overflow. ``None``
    means no bound was recovered. The gap and verification tolerances are checked
    internally before this reporting conversion.
    """

    verified: bool
    primal_valid: bool
    dual_valid: bool
    stationarity_inf: float
    complementarity_inf: float
    gap: float
    reason: str
    dual_bound: float | None = None
    domain_refined: bool = False
    recovery_diagnostics: dict | None = None


def _original_dual_bound(linear, exact_dual):
    if exact_dual is None:
        return None
    lower = linear.objective_sense is ObjectiveSense.MINIMIZE
    exact = (exact_dual if lower else -exact_dual) + rational(linear.objective_offset)
    rounded = bounded_float(exact)
    outward = -np.inf if lower else np.inf
    if np.isinf(rounded):
        # Overflow toward the unsafe side must saturate to the largest finite
        # float instead: +inf is not a lower bound on a finite optimum.
        return float(np.nextafter(rounded, outward)) if rounded != outward else rounded
    if (lower and rational(rounded) > exact) or (not lower and rational(rounded) < exact):
        with np.errstate(over='ignore', under='ignore'):
            rounded = float(np.nextafter(rounded, outward))
    return rounded


def verify_optimality(problem, x, dual, *, tolerances=None, fast_reject=False,
                      extended_recovery=False, recovery_options=None, proof_policy=None, workspace=None, curvature_witness=None) -> OptimalityCheck:
    # Public verification always checks the supplied candidate independently.
    from .workspace import verification_scope, VerificationLimit
    try:
        with verification_scope(proof_policy,workspace):
            return _verify_optimality_with_primal_report(
                problem, x, dual, None, tolerances=tolerances, fast_reject=fast_reject,
                extended_recovery=extended_recovery, recovery_options=recovery_options,
                curvature_witness=curvature_witness,
            )
    except VerificationLimit as exc:
        return OptimalityCheck(False,False,False,np.inf,np.inf,np.inf,str(exc))


@controlled_verification
def _verify_optimality_with_primal_report(
    problem, x, dual, primal_report: ValidationReport | None, *, tolerances=None,
    fast_reject=False, extended_recovery=False, recovery_options=None,
    curvature_witness=None,
) -> OptimalityCheck:
    """Internal reuse of a report computed for this candidate and tolerances.

    Only the executor supplies a report, from its own immediate validation.
    It must exclude any inconsistent backend-reported objective: the public
    certificate contract checks the canonical candidate, not that claim.
    ``None`` selects the self-contained public verification path.
    """
    call_start = perf_counter()
    options = dict(recovery_options or {})
    limit = options.get('time_limit_s', 10.) if extended_recovery else None
    if limit is not None and (isinstance(limit, bool) or not np.isfinite(limit) or limit <= 0):
        raise ValueError('recovery time limit must be finite and positive')
    tol = tolerances or ValidationTolerances()
    bad = lambda reason: OptimalityCheck(False, False, False, np.inf, np.inf, np.inf, reason)
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if not isinstance(linear, LinearProblem) or linear.has_integer_variables:
        return bad('only continuous LP and convex QP are eligible')
    try:
        x, y = np.asarray(x, dtype=float), np.asarray(dual, dtype=float)
    except (TypeError, ValueError, OverflowError):
        return bad("certificate is not numeric")
    # LP bounds need the same products as [A; I], but do not need a new sparse
    # identity/stack/transpose on every call. Keep the QP certificate path intact.
    B = (sparse.vstack([linear.A, sparse.eye(linear.n_variables)], format='csr')
         if isinstance(problem, QuadraticProblem) else None)
    lo, hi = np.r_[linear.constraint_lower, linear.variable_lower], np.r_[linear.constraint_upper, linear.variable_upper]
    if x.shape != (linear.n_variables,) or y.shape != lo.shape or not np.isfinite(x).all() or not np.isfinite(y).all():
        return bad('certificate shape or finite-value check failed')
    primal = (validate_solution(problem, CandidateSolution(x), tolerances=tol)
              if primal_report is None else primal_report).valid
    if fast_reject and not primal:
        return bad('primal candidate outside tolerances')
    sign = 1. if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.
    px = problem.P@x if isinstance(problem, QuadraticProblem) else np.zeros_like(x)
    if isinstance(problem, QuadraticProblem):
        from solverpilot.problem.quadratic import require_confirmed_convexity
        if curvature_witness is None: require_confirmed_convexity(problem)
        from solverpilot.problem.curvature import certify_psd
        if curvature_witness is None: curvature = certify_psd(problem.P)
        else:
            from solverpilot.problem.curvature import certify_psd_factor
            curvature=certify_psd_factor(problem.P,**dict(curvature_witness))
        if not curvature.certified:
            return bad("independent PSD evidence unavailable: " + curvature.reason)
    bound = np.where(y > 0, hi, lo)
    active = y != 0
    if not np.isfinite(bound[active]).all():
        return bad('dual multiplier points toward an infinite bound')
    with np.errstate(over='ignore', invalid='ignore'):
        terms = np.zeros_like(y); terms[active] = y[active]*bound[active]
        if B is None:
            row_y, bound_y = y[:linear.n_constraints], y[linear.n_constraints:]
            dual_activity = linear.A.T@row_y + bound_y
            activity = np.r_[linear.A@x, x]
            dual_scale = abs(linear.A.T)@abs(row_y) + abs(bound_y)
        else:
            dual_activity, activity = B.T@y, B@x
            dual_scale = abs(B.T)@abs(y)
        stationarity = px + sign*linear.c + dual_activity
        comp = np.zeros_like(y); comp[active] = y[active]*(bound[active]-activity[active])
        primal_value = float(.5*x@px + sign*linear.c@x)
        st = float(np.max(np.abs(stationarity), initial=0))
        cp = float(np.max(np.abs(comp), initial=0))
        scale = np.maximum(1., np.abs(px)+np.abs(linear.c)+dual_scale)
        st_ok = bool(np.all(np.abs(stationarity) <= tol.feasibility+tol.feasibility_rel*scale))
        if fast_reject and (not np.isfinite([st, cp]).all() or not np.isfinite(scale).all() or not st_ok):
            return OptimalityCheck(False, True, False, st, cp, np.inf,
                                   'stationarity outside tolerances')
        # A small stationarity residual can have a large effect on a wide box.
        # The tangent Lagrangian gives a valid lower bound after minimizing
        # this residual over the variable domain. Exact binary64 arithmetic
        # avoids losing a residual through cancellation in the sparse product.
        if isinstance(problem, QuadraticProblem):
            exact_px = matvec(problem.P, x)
            residual = [p + rational(sign*c) + b
                        for p, c, b in zip(exact_px, linear.c, matvec(B.T, y))]
            half_xpx = sum((rational(v)*p for v, p in zip(x, exact_px)), rational(0))/2
            bound_dual, bound_values = y[active], bound[active]
        else:
            # Keep variable bounds in the domain, not also in the Lagrangian.
            # A rounded bound multiplier can turn an exactly nonnegative
            # reduced cost into a tiny negative residual on a half-infinite
            # variable. That needlessly makes the full-dual box bound infinite.
            # For every feasible z, y_bound*(z-bound) <= 0, so omitting these
            # terms gives a valid, at least as strong lower bound. The supplied
            # full dual still undergoes all sign/KKT/complementarity checks.
            row_dual = y[:linear.n_constraints]
            row_active = row_dual != 0
            residual = [rational(sign*c) + b
                        for c, b in zip(linear.c, matvec(linear.A.T, row_dual))]
            half_xpx = rational(0)
            bound_dual = row_dual[row_active]
            bound_values = bound[:linear.n_constraints][row_active]
        correction = box_min(residual, linear.variable_lower, linear.variable_upper)
        domain_refined = False
        recovery_diagnostics = None
        if correction is None and isinstance(problem, LinearProblem):
            from .lp_dual import implied_lp_box, row_residual_lower_bound
            lower, upper = implied_lp_box(problem)
            correction = box_min(residual, lower, upper)
            if correction is None:
                correction = row_residual_lower_bound(problem, residual, lower, upper)
            if correction is None:
                from .lp_dual import dual_slack_residual_lower_bound, equality_residual_lower_bound
                basis = options.pop('basis', None) if extended_recovery else None
                if extended_recovery:
                    options['time_limit_s'] = None if limit is None else max(1e-12, limit-(perf_counter()-call_start))
                else:
                    # A bounded equality-only stage closes coupled residuals
                    # without invoking another solver or treating inequalities
                    # as equations. Larger/slack recovery remains opt-in.
                    options = dict(max_pivots=256, max_visits=50000, max_bits=8192)
                recovery_diagnostics = {'mode': 'extended' if extended_recovery else 'bounded_equality', 'equality': {}}
                correction = equality_residual_lower_bound(problem, residual, lower, upper,
                    diagnostics=recovery_diagnostics['equality'], **options)
                if correction is None and extended_recovery:
                    remaining = None if limit is None else limit-(perf_counter()-call_start)
                    if remaining is None or remaining > 0:
                        options['time_limit_s'] = remaining
                        recovery_diagnostics['dual_slack'] = {}
                        # Residual/base above omit bound multipliers. Recovery
                        # must use that same Lagrangian, including zero penalties
                        # for the variable-bound rows it adds to its slack model.
                        recovery_dual = np.r_[y[:linear.n_constraints], np.zeros(linear.n_variables)]
                        correction = dual_slack_residual_lower_bound(problem, residual, lower, upper, recovery_dual, x=x, basis=basis,
                            diagnostics=recovery_diagnostics['dual_slack'], **options)
            domain_refined = correction is not None
        exact_primal = half_xpx + dot(sign*linear.c, x)
        exact_dual = None if correction is None else -half_xpx-dot(bound_dual, bound_values)+correction
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
        allowed = tol.objective_abs+tol.objective_rel*max(1., abs(primal_value), abs(dual_value))
    finite = bool(np.isfinite([st, cp, gap, allowed]).all() and np.isfinite(scale).all())
    dual_ok = finite and st_ok
    verified = primal and dual_ok and gap <= allowed and cp <= allowed
    late = limit is not None and perf_counter()-call_start >= limit
    if late: verified = False
    return OptimalityCheck(verified, primal, dual_ok, st, cp, gap,
                           'numerically verified KKT and domain-corrected gap' if verified else
                           ('recovery time limit exceeded' if late else
                            'stationarity error has no finite lower bound' if exact_dual is None else 'certificate outside tolerances'),
                           _original_dual_bound(linear, exact_dual), domain_refined, recovery_diagnostics)


def verify_infeasibility(problem: LinearProblem, dual, *, atol=1e-8) -> bool:
    """Check a normalized Farkas separator for the continuous relaxation."""
    if not np.isfinite(atol) or atol < 0:
        raise ValueError('atol must be finite and nonnegative')
    B = sparse.vstack([problem.A, sparse.eye(problem.n_variables)], format='csr')
    lo, hi = np.r_[problem.constraint_lower, problem.variable_lower], np.r_[problem.constraint_upper, problem.variable_upper]
    try:
        y = np.asarray(dual, dtype=float)
    except (TypeError, ValueError, OverflowError):
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
    except (TypeError, ValueError, OverflowError):
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
