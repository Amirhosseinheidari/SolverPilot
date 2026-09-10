from __future__ import annotations

import numpy as np

from solverpilot.problem import LinearProblem, QuadraticProblem, VariableDomain

from .result import CandidateSolution, ValidationReport, ValidationTolerances

_FEASIBILITY_REL = 1e-9


def _max_violation(values: np.ndarray, lower: np.ndarray, upper: np.ndarray) -> float:
    if not np.isfinite(values).all():
        return float("inf")
    below = np.maximum(lower - values, 0.0)
    above = np.maximum(values - upper, 0.0)
    v = np.maximum(below, above)
    return float(np.max(v)) if v.size else 0.0


def _scaled_feasibility(
    values: np.ndarray,
    lower: np.ndarray,
    upper: np.ndarray,
    *,
    atol: float,
    rtol: float,
    extra_scale: np.ndarray | None = None,
) -> tuple[float, bool]:
    """Return max normalized violation and whether every row is within tolerance.

    The absolute tolerance remains the numerical floor.  The relative term scales
    against the magnitude of the candidate, finite bounds and (for constraints)
    an independently supplied row/activity scale.
    """
    if not np.isfinite(values).all():
        return float("inf"), False
    scale = np.maximum(1.0, np.abs(values))
    if extra_scale is not None:
        if not np.isfinite(extra_scale).all():
            return float("inf"), False
        scale = np.maximum(scale, np.asarray(extra_scale, dtype=np.float64))
    max_normalized = 0.0
    ok = True
    # Each side uses its own bound: a distant upper bound must not relax a
    # violated lower bound (or vice versa).
    for bound, direction in ((lower, -1.0), (upper, 1.0)):
        mask = np.isfinite(bound)
        with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
            violation = np.maximum(direction * (values[mask] - bound[mask]), 0.0)
            allowed = float(atol) + float(rtol) * np.maximum(scale[mask], np.abs(bound[mask]))
            normalized = np.divide(violation, allowed, out=np.full_like(violation, np.inf), where=allowed > 0)
        normalized[violation == 0] = 0.0
        if normalized.size:
            max_normalized = max(max_normalized, float(np.max(normalized)))
        ok = ok and bool(np.all(np.isfinite(violation) & np.isfinite(allowed) & (violation <= allowed)))
    return max_normalized, ok


def _linear_objective(problem: LinearProblem, x: np.ndarray) -> float:
    return float(problem.c @ x + problem.objective_offset)


def _quadratic_objective(problem: QuadraticProblem, x: np.ndarray) -> float:
    return float(0.5 * x @ (problem.P @ x) + problem.linear.c @ x + problem.linear.objective_offset)


def validate_solution(
    problem: LinearProblem | QuadraticProblem,
    solution: CandidateSolution,
    *,
    tolerances: ValidationTolerances | None = None,
) -> ValidationReport:
    tol = ValidationTolerances() if tolerances is None else tolerances
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    x = np.asarray(solution.x, dtype=np.float64)

    if x.shape != (linear.n_variables,):
        raise ValueError(f"solution x must have shape ({linear.n_variables},), got {x.shape}")
    if not np.isfinite(x).all():
        return ValidationReport(
            valid=False,
            max_bound_violation=float("inf"),
            max_constraint_violation=float("inf"),
            max_integrality_violation=float("inf") if linear.has_integer_variables else None,
            objective_recomputed=None,
            objective_reported=solution.objective_reported,
            objective_difference=None,
            objective_consistent=False if solution.objective_reported is not None else None,
            warnings=("solution contains NaN or infinity",),
        )

    bound_v = _max_violation(x, linear.variable_lower, linear.variable_upper)
    bound_scaled_v, bounds_ok = _scaled_feasibility(
        x,
        linear.variable_lower,
        linear.variable_upper,
        atol=tol.feasibility,
        rtol=_FEASIBILITY_REL,
    )

    activity = np.asarray(linear.A @ x, dtype=np.float64).reshape(-1)
    constraint_v = _max_violation(activity, linear.constraint_lower, linear.constraint_upper)
    row_scale = np.asarray(np.abs(linear.A) @ np.abs(x), dtype=np.float64).reshape(-1)
    constraint_scaled_v, constraints_ok = _scaled_feasibility(
        activity,
        linear.constraint_lower,
        linear.constraint_upper,
        atol=tol.feasibility,
        rtol=_FEASIBILITY_REL,
        extra_scale=row_scale,
    )

    integer_mask = np.isin(
        linear.domains,
        [VariableDomain.INTEGER.value, VariableDomain.BINARY.value],
    )
    if np.any(integer_mask):
        int_v = float(np.max(np.abs(x[integer_mask] - np.rint(x[integer_mask]))))
    else:
        int_v = None

    objective = (
        _quadratic_objective(problem, x)
        if isinstance(problem, QuadraticProblem)
        else _linear_objective(problem, x)
    )

    reported = solution.objective_reported
    if not np.isfinite(objective):
        objective_diff = None if reported is None else float("inf")
        objective_ok = False
    elif reported is None:
        objective_diff = None
        objective_ok = None
    elif not np.isfinite(reported):
        objective_diff = float("inf")
        objective_ok = False
    else:
        objective_diff = float(abs(objective - reported))
        objective_ok = bool(
            objective_diff
            <= tol.objective_abs + tol.objective_rel * max(abs(objective), abs(reported))
        )

    valid = (
        bounds_ok
        and constraints_ok
        and (int_v is None or int_v <= tol.integrality)
        and (objective_ok is not False)
    )

    warnings: list[str] = []
    if not np.isfinite(objective):
        warnings.append("canonical objective is non-finite")
    if not bounds_ok:
        warnings.append("variable bound violation")
    elif bound_v > tol.feasibility:
        warnings.append("variable bound residual accepted by scale-aware feasibility tolerance")
    if not constraints_ok:
        warnings.append("constraint violation")
    elif constraint_v > tol.feasibility:
        warnings.append("constraint residual accepted by scale-aware feasibility tolerance")
    if int_v is not None and int_v > tol.integrality:
        warnings.append("integrality violation")
    if objective_ok is False:
        warnings.append("reported objective is inconsistent with canonical objective")

    return ValidationReport(
        valid=valid,
        max_bound_violation=bound_v,
        max_constraint_violation=constraint_v,
        max_integrality_violation=int_v,
        objective_recomputed=objective,
        objective_reported=reported,
        objective_difference=objective_diff,
        objective_consistent=objective_ok,
        warnings=tuple(warnings),
    )
