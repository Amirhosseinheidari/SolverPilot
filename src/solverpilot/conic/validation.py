from __future__ import annotations

from dataclasses import dataclass
import math
from decimal import Decimal, localcontext
from fractions import Fraction
import numpy as np

from solverpilot.validate.core import _max_violation, _scaled_feasibility

from .ir import ConeKind, ConicProblem
from solverpilot.validate._certificate_arithmetic import rational, bounded_float


@dataclass(frozen=True, slots=True)
class ConeCheck:
    source_id: str | None
    kind: ConeKind
    valid: bool
    violation: float
    detail: str


@dataclass(frozen=True, slots=True)
class ConicValidationReport:
    valid: bool
    max_variable_violation: float
    max_linear_violation: float
    max_cone_violation: float
    cone_checks: tuple[ConeCheck, ...]
    objective: float | None


def _scaled_tol(atol: float, rtol: float, magnitude: float) -> float:
    return float(atol + rtol * max(1.0, abs(float(magnitude))))


def _strict_log_member(heads, weights, tail, tail_squared=None, tolerance=0.):
    """Conservative log-domain check without normalizing positive heads to zero."""
    if any(v < 0 for v in heads):
        return False
    square = rational(tail)**2 if tail_squared is None else tail_squared
    tol = rational(tolerance)
    if square <= tol**2:
        return True
    if any(v == 0 for v in heads):
        return False
    exact_weights = tuple(w if isinstance(w, Fraction) else rational(w) for w in weights)
    if all(v == heads[0] for v in heads) and sum(exact_weights, Fraction()) == 1:
        return square <= (rational(heads[0])+tol)**2
    if len(heads) == 2 and tuple(weights) == (.5, .5):
        product = rational(heads[0])*rational(heads[1])
        difference = square-product-tol**2
        return difference <= 0 or difference**2 <= 4*product*tol**2
    with localcontext() as context:
        context.prec = 96
        product_log = sum(((Decimal(w.numerator)/Decimal(w.denominator))*Decimal.from_float(float(v)).ln()
                           for w, v in zip(exact_weights, heads)), Decimal(0))
        relaxed_tail = (Decimal(square.numerator)/Decimal(square.denominator)).sqrt()-Decimal.from_float(float(tolerance))
        tail_log = relaxed_tail.ln()
        # Refuse an unresolved transcendental boundary rather than turn rounded
        # equality into membership. Equal-head and rational square cases above
        # retain their exactly decidable boundaries.
        uncertainty = Decimal('1e-90')*max(Decimal(1), abs(product_log), abs(tail_log))*(len(heads)+4)
        return tail_log < product_log-uncertainty


def _strict_exponential_member(a, b, c, tolerance=0.):
    if b == 0:
        return a <= tolerance and c >= -tolerance
    if b < 0 or c <= 0:
        return False
    if c >= b and a <= tolerance:
        return True
    with localcontext() as context:
        context.prec = 96
        da, db, dc = (Decimal.from_float(v) for v in (a, b, c))
        left = (da-Decimal.from_float(float(tolerance)))/db
        bound = dc.ln()-db.ln()
        uncertainty = Decimal('1e-90')*max(Decimal(1), abs(left), abs(bound), abs(dc.ln()), abs(db.ln()))
        return left < bound-uncertainty


def _componentwise_feasibility(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    atol: float,
    rtol: float,
    extra_scale: np.ndarray | None = None,
) -> tuple[float, bool]:
    _, ok = _scaled_feasibility(values, lower, upper, atol=atol, rtol=rtol, extra_scale=extra_scale)
    return _max_violation(values, lower, upper), ok


def _psd_check(M: np.ndarray, *, atol: float, rtol: float) -> tuple[bool, float, float, float]:
    """Check symmetry and PSD after local congruence scaling.

    The congruence transform preserves inertia, while removing the dangerous
    dependence on an unrelated huge matrix entry that a single global norm creates.
    """
    if M.size == 0:
        return True, 0.0, 0.0, 0.0
    abs_m = np.abs(M)
    sym_allowed = float(atol) + float(rtol) * np.maximum(1.0, np.maximum(abs_m, abs_m.T))
    asym_matrix = np.abs(M - M.T)
    asym = float(np.max(asym_matrix))
    symmetry_ok = bool(np.all(asym_matrix <= sym_allowed))
    sym = 0.5 * M + 0.5 * M.T
    eigmin = float(np.linalg.eigvalsh(sym)[0])

    row_scale = np.maximum(1.0, np.max(np.abs(sym), axis=1))
    inv_sqrt = 1.0 / np.sqrt(row_scale)
    scaled = inv_sqrt[:, None] * sym * inv_sqrt[None, :]
    scaled_eigmin = float(np.linalg.eigvalsh(scaled)[0])
    scaled_norm = float(np.linalg.norm(scaled, ord=np.inf))
    psd_tol = float(atol) + float(rtol) * max(1.0, scaled_norm)
    return bool(symmetry_ok and scaled_eigmin >= -psd_tol), asym, eigmin, scaled_eigmin


def _cone_check(kind: ConeKind, value: np.ndarray, *, atol: float, rtol: float, source_id: str | None, alpha: float | None = None, weights=None) -> ConeCheck:
    if not np.isfinite(value).all():
        return ConeCheck(source_id, kind, False, float("inf"), "non-finite cone activity")
    if kind is ConeKind.GENERALIZED_POWER:
        y = np.asarray(value, dtype=float).reshape(-1)
        if weights is None:
            raise ValueError("generalized power weights required")
        count = len(weights)
        heads, tail = y[:count], y[count:]
        if any(v < -_scaled_tol(atol, rtol, v) for v in heads):
            return ConeCheck(source_id, kind, False, float("inf"), "negative generalized power coordinate")
        norm = math.hypot(*tail)
        try:
            product = math.exp(math.fsum(float(w)*math.log(float(v)) for w,v in zip(weights,heads))) if np.all(heads > 0) else 0.
        except OverflowError:
            product = float(np.max(heads))
        violation = max(0., norm-product, -float(np.min(heads)))
        tolerance = _scaled_tol(atol, rtol, max(norm, product))
        ok = math.isfinite(norm) and math.isfinite(tolerance) and violation <= tolerance
        if ok and np.all(heads >= 0):
            ok = _strict_log_member(heads, weights, norm,
                sum((rational(v)**2 for v in tail), Fraction()), tolerance)
        if atol == rtol == 0:
            ok = _strict_log_member(heads, weights, norm,
                sum((rational(v)**2 for v in tail), Fraction()))
        if not ok and violation == 0:
            violation = float(np.nextafter(0., np.inf))
        return ConeCheck(source_id, kind, ok, violation,
                         f"original_gap={violation}, original_tolerance={tolerance}")
    if kind in (ConeKind.EXPONENTIAL, ConeKind.POWER):
        scale = max(1., float(np.max(np.abs(value))))
        a, b, c = (float(v) for v in np.asarray(value).reshape(-1))
        tol = _scaled_tol(atol, rtol, scale)
        if kind is ConeKind.EXPONENTIAL:
            if value.reshape(-1)[1] < -_scaled_tol(atol, rtol, value.reshape(-1)[1]) or value.reshape(-1)[2] < -_scaled_tol(atol, rtol, value.reshape(-1)[2]):
                return ConeCheck(source_id, kind, False, float("inf"), "negative exponential cone coordinate")
            if b > 0 and c > 0:
                violation = max(0., a-b*(math.log(c)-math.log(b)))
            elif abs(b) <= tol:
                violation = max(0., a, -c, abs(b))
            else:
                violation = float('inf')
        else:
            if alpha is None or not 0 < alpha < 1:
                raise ValueError('valid power exponent required')
            if any(v < -_scaled_tol(atol, rtol, v) for v in value.reshape(-1)[:2]):
                return ConeCheck(source_id, kind, False, float('inf'), 'negative power cone coordinate')
            try:
                product = math.exp(alpha*math.log(a)+(1-alpha)*math.log(b)) if a > 0 and b > 0 else 0.
            except OverflowError:
                product = max(a, b)
            violation = max(0., -a, -b, abs(c)-product)
        ok = math.isfinite(tol) and violation <= tol
        if ok and kind is ConeKind.EXPONENTIAL and b > 0 and c > 0:
            ok = _strict_exponential_member(a, b, c, tol)
        elif ok and kind is ConeKind.POWER and a >= 0 and b >= 0:
            ok = _strict_log_member((a, b), (rational(alpha), 1-rational(alpha)), abs(c), tolerance=tol)
        if atol == rtol == 0:
            ok = (_strict_exponential_member(a, b, c) if kind is ConeKind.EXPONENTIAL
                  else _strict_log_member((a, b), (rational(alpha), 1-rational(alpha)), abs(c)))
        if not ok and violation == 0:
            violation = float(np.nextafter(0., np.inf))
        return ConeCheck(source_id, kind, bool(ok), float(violation),
                         f'original_violation={violation}, original_tolerance={tol}')
    if kind is ConeKind.SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        t = float(y[0]); norm = math.hypot(*y[1:])
        viol = max(0.0, norm - t)
        tol = _scaled_tol(atol, rtol, max(abs(t), norm))
        norm2 = sum((rational(v)**2 for v in y[1:]), Fraction())
        relaxed_head = rational(t)+rational(tol) if math.isfinite(tol) else Fraction(-1)
        ok = math.isfinite(norm) and math.isfinite(tol) and relaxed_head >= 0 and relaxed_head**2 >= norm2
        if not ok and viol == 0:
            viol = float(np.nextafter(0., np.inf))
        return ConeCheck(source_id, kind, ok, viol, f"norm={norm:.17g}, t={t:.17g}, tol={tol:.3g}; exact squared membership")
    if kind is ConeKind.ROTATED_SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        u = float(y[0]); v = float(y[1])
        norm2 = sum((rational(v)**2 for v in y[2:]), Fraction())
        product = 2*rational(u)*rational(v)
        gap = norm2 - product
        tol = rational(atol)+rational(rtol)*max(Fraction(1), norm2, abs(product))
        nonnegative = u >= -_scaled_tol(atol, rtol, u) and v >= -_scaled_tol(atol, rtol, v)
        viol = max(0.0, -u, -v, bounded_float(max(Fraction(), gap)))
        return ConeCheck(source_id, kind, nonnegative and gap <= tol, viol,
                         f"u={u:.17g}, v={v:.17g}; exact original squared gap")
    if kind is ConeKind.POSITIVE_SEMIDEFINITE:
        M = np.asarray(value, dtype=float)
        ok, asym, eigmin, scaled_eigmin = _psd_check(M, atol=atol, rtol=rtol)
        if atol == rtol == 0:
            from solverpilot.problem.curvature import certify_psd
            certificate = certify_psd(M)
            ok = certificate.certified and asym == 0
            if not certificate.certified:
                return ConeCheck(source_id, kind, False, max(asym, max(0.,-eigmin)),
                                 f'strict PSD membership not established: {certificate.status}/{certificate.method}')
        viol = max(asym, max(0.0, -eigmin))
        return ConeCheck(
            source_id,
            kind,
            ok,
            viol,
            f"asym={asym:.3g}, eigmin={eigmin:.17g}, scaled_eigmin={scaled_eigmin:.17g}",
        )
    raise AssertionError(kind)


def validate_conic_solution(
    problem: ConicProblem,
    x: np.ndarray | None,
    *,
    atol: float = 1e-7,
    rtol: float = 1e-7,
) -> ConicValidationReport:
    for name, raw in (("atol", atol), ("rtol", rtol)):
        if isinstance(raw, bool) or type(raw) not in (int, float) or not np.isfinite(float(raw)) or float(raw) < 0:
            raise ValueError(f"{name} must be finite and non-negative")
    atol = float(atol)
    rtol = float(rtol)
    if x is None:
        return ConicValidationReport(False, float("inf"), float("inf"), float("inf"), (), None)
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    if x.shape != (problem.n_variables,) or not np.all(np.isfinite(x)):
        return ConicValidationReport(False, float("inf"), float("inf"), float("inf"), (), None)

    var_viol, vars_ok = _componentwise_feasibility(
        x,
        problem.variable_lower,
        problem.variable_upper,
        atol=atol,
        rtol=rtol,
    )
    ax = np.asarray(problem.A @ x, dtype=float).reshape(-1)
    row_scale = np.asarray(np.abs(problem.A) @ np.abs(x), dtype=np.float64).reshape(-1)
    lin_viol, linear_ok = _componentwise_feasibility(
        ax,
        problem.constraint_lower,
        problem.constraint_upper,
        atol=atol,
        rtol=rtol,
        extra_scale=row_scale,
    )
    checks = tuple(_cone_check(c.kind, c.value(x), atol=atol, rtol=rtol, source_id=c.source_id, alpha=c.metadata.get("alpha"), weights=c.metadata.get("weights")) for c in problem.cones)
    cone_viol = max((c.violation for c in checks), default=0.0)
    objective = problem.objective_value(x)
    valid = bool(vars_ok and linear_ok and all(c.valid for c in checks) and np.isfinite(objective))
    return ConicValidationReport(valid, var_viol, lin_viol, cone_viol, checks, objective)
