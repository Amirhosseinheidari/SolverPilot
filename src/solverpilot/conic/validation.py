from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np

from solverpilot.validate.core import _max_violation, _scaled_feasibility

from .ir import ConeKind, ConicProblem


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


def _cone_check(kind: ConeKind, value: np.ndarray, *, atol: float, rtol: float, source_id: str | None, alpha: float | None = None) -> ConeCheck:
    if not np.isfinite(value).all():
        return ConeCheck(source_id, kind, False, float("inf"), "non-finite cone activity")
    if kind in (ConeKind.EXPONENTIAL, ConeKind.POWER):
        scale = max(1., float(np.max(np.abs(value))))
        a, b, c = np.asarray(value).reshape(-1)/scale
        tol = atol/scale + rtol
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
            product = math.exp(alpha*math.log(a)+(1-alpha)*math.log(b)) if a > 0 and b > 0 else 0.
            violation = max(0., -a, -b, abs(c)-product)
        return ConeCheck(source_id, kind, bool(violation <= tol), float(violation*scale),
                         f'normalized_violation={violation}, normalized_tolerance={tol}')
    if kind is ConeKind.SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        t = float(y[0]); norm = math.hypot(*y[1:])
        viol = max(0.0, norm - t)
        tol = _scaled_tol(atol, rtol, max(abs(t), norm))
        return ConeCheck(source_id, kind, math.isfinite(norm) and math.isfinite(tol) and viol <= tol, viol, f"norm={norm:.17g}, t={t:.17g}, tol={tol:.3g}")
    if kind is ConeKind.ROTATED_SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        u = float(y[0]); v = float(y[1])
        scale = max(1.0, float(np.max(np.abs(y))))
        z = y / scale
        norm2 = float(np.dot(z[2:], z[2:]))
        product = 2.0 * float(z[0]) * float(z[1])
        gap = norm2 - product
        # Compare the squared inequality in normalized coordinates, avoiding
        # inf - inf. Nonnegativity uses separate, unsquared tolerances.
        tol = (atol / scale) / scale + rtol * max((1.0 / scale) / scale, norm2, abs(product))
        nonnegative = u >= -_scaled_tol(atol, rtol, u) and v >= -_scaled_tol(atol, rtol, v)
        viol = max(0.0, -u, -v, (max(0.0, gap) * scale) * scale)
        return ConeCheck(source_id, kind, nonnegative and gap <= tol, viol, f"u={u:.17g}, v={v:.17g}, normalized_gap={gap:.17g}, tol={tol:.3g}")
    if kind is ConeKind.POSITIVE_SEMIDEFINITE:
        M = np.asarray(value, dtype=float)
        ok, asym, eigmin, scaled_eigmin = _psd_check(M, atol=atol, rtol=rtol)
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
    checks = tuple(_cone_check(c.kind, c.value(x), atol=atol, rtol=rtol, source_id=c.source_id, alpha=c.metadata.get("alpha")) for c in problem.cones)
    cone_viol = max((c.violation for c in checks), default=0.0)
    objective = problem.objective_value(x)
    valid = bool(vars_ok and linear_ok and all(c.valid for c in checks) and np.isfinite(objective))
    return ConicValidationReport(valid, var_viol, lin_viol, cone_viol, checks, objective)
