from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Sequence

from .distance import route_cost
from .model import TSPInstance
from .solution import TSPSolution


@dataclass(frozen=True, slots=True)
class TSPValidationReport:
    valid: bool
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    objective_recomputed: float | None
    objective_reported: float | None
    objective_difference: float | None


def validate_tsp_route(
    instance: TSPInstance,
    route_or_solution: Sequence[int] | TSPSolution,
    *,
    reported_objective: float | None = None,
    atol: float = 1e-9,
    rtol: float = 1e-9,
) -> TSPValidationReport:
    if isinstance(atol, bool) or isinstance(rtol, bool):
        raise ValueError("TSP validation tolerances must be non-negative finite numbers")
    atol = float(atol); rtol = float(rtol)
    if not isfinite(atol) or not isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("TSP validation tolerances must be non-negative finite numbers")
    errors: list[str] = []
    warnings: list[str] = []
    if isinstance(route_or_solution, TSPSolution):
        route = tuple(route_or_solution.route)
        if reported_objective is None:
            reported_objective = route_or_solution.objective
    else:
        try:
            route = tuple(route_or_solution)
        except TypeError:
            return TSPValidationReport(False, ("route must be iterable",), (), None, reported_objective, None)
    if len(route) != instance.n_nodes + 1:
        errors.append(f"route must contain n_nodes + 1 entries ({instance.n_nodes + 1})")
    normalized: list[int] = []
    for value in route:
        if isinstance(value, bool) or not isinstance(value, int):
            errors.append("route entries must be integer node indices")
            continue
        normalized.append(int(value))
    if len(normalized) == len(route):
        invalid = sorted({x for x in normalized if x < 0 or x >= instance.n_nodes})
        if invalid:
            errors.append(f"route contains out-of-range node indices: {invalid}")
        if normalized and normalized[0] != normalized[-1]:
            errors.append("route must be closed")
        if not invalid and len(normalized) >= 2:
            interior = normalized[:-1]
            counts = Counter(interior)
            missing = sorted(set(range(instance.n_nodes)) - set(interior))
            duplicated = sorted(k for k, v in counts.items() if v != 1)
            if missing:
                errors.append(f"route is missing nodes: {missing}")
            if duplicated:
                errors.append(f"route does not visit nodes exactly once: {duplicated}")
    recomputed: float | None = None
    if not errors:
        recomputed = route_cost(instance, normalized)
    reported: float | None = None
    difference: float | None = None
    if reported_objective is not None:
        if isinstance(reported_objective, bool):
            errors.append("reported objective must be numeric, not bool")
            reported_objective = None
        try:
            reported = None if reported_objective is None else float(reported_objective)
        except (TypeError, ValueError):
            errors.append("reported objective must be numeric")
        else:
            if reported is None:
                pass
            elif not isfinite(reported):
                errors.append("reported objective must be finite")
            elif recomputed is not None:
                difference = reported - recomputed
                if not isclose(reported, recomputed, rel_tol=rtol, abs_tol=atol):
                    errors.append("reported objective does not match independently recomputed route cost")
    return TSPValidationReport(not errors, tuple(errors), tuple(warnings), recomputed, reported, difference)
