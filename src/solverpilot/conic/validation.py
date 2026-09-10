from __future__ import annotations

from dataclasses import dataclass
import numpy as np

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
    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    violation = np.maximum(below, above)
    finite_lower = np.where(np.isfinite(lower), np.abs(lower), 0.0)
    finite_upper = np.where(np.isfinite(upper), np.abs(upper), 0.0)
    scale = np.maximum.reduce([np.ones_like(values), np.abs(values), finite_lower, finite_upper])
    if extra_scale is not None:
        scale = np.maximum(scale, np.asarray(extra_scale, dtype=np.float64))
    allowed = float(atol) + float(rtol) * scale
    return (float(np.max(violation)) if violation.size else 0.0, bool(np.all(violation <= allowed)))


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
    sym = 0.5 * (M + M.T)
    eigmin = float(np.linalg.eigvalsh(sym)[0])

    row_scale = np.maximum(1.0, np.max(np.abs(sym), axis=1))
    inv_sqrt = 1.0 / np.sqrt(row_scale)
    scaled = inv_sqrt[:, None] * sym * inv_sqrt[None, :]
    scaled_eigmin = float(np.linalg.eigvalsh(scaled)[0])
    scaled_norm = float(np.linalg.norm(scaled, ord=np.inf))
    psd_tol = float(atol) + float(rtol) * max(1.0, scaled_norm)
    return bool(symmetry_ok and scaled_eigmin >= -psd_tol), asym, eigmin, scaled_eigmin


def _cone_check(kind: ConeKind, value: np.ndarray, *, atol: float, rtol: float, source_id: str | None) -> ConeCheck:
    if kind is ConeKind.SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        t = float(y[0]); norm = float(np.linalg.norm(y[1:]))
        viol = max(0.0, norm - t)
        tol = _scaled_tol(atol, rtol, max(abs(t), norm))
        return ConeCheck(source_id, kind, viol <= tol, viol, f"norm={norm:.17g}, t={t:.17g}, tol={tol:.3g}")
    if kind is ConeKind.ROTATED_SECOND_ORDER:
        y = np.asarray(value, dtype=float).reshape(-1)
        u = float(y[0]); v = float(y[1]); norm2 = float(np.dot(y[2:], y[2:]))
        cone_gap = norm2 - 2.0 * u * v
        viol = max(0.0, -u, -v, cone_gap)
        tol = _scaled_tol(atol, rtol, max(abs(u), abs(v), norm2, abs(2*u*v)))
        return ConeCheck(source_id, kind, viol <= tol, viol, f"u={u:.17g}, v={v:.17g}, norm2={norm2:.17g}, tol={tol:.3g}")
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
    checks = tuple(_cone_check(c.kind, c.value(x), atol=atol, rtol=rtol, source_id=c.source_id) for c in problem.cones)
    cone_viol = max((c.violation for c in checks), default=0.0)
    valid = bool(vars_ok and linear_ok and all(c.valid for c in checks))
    return ConicValidationReport(valid, var_viol, lin_viol, cone_viol, checks, problem.objective_value(x))
