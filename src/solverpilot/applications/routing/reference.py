from __future__ import annotations

from dataclasses import dataclass
from itertools import permutations
from math import comb, factorial
from time import perf_counter

from .errors import VRPReferenceLimitError
from .model import VRPInstance
from .solution import VRPRoute, VRPSolution, _INDEPENDENT_PROOF_TOKEN
from .validation import route_distance, validate_vrp_solution


@dataclass(frozen=True, slots=True)
class VRPReferenceResult:
    solution: VRPSolution | None
    evaluations: int
    search_space_upper_bound: int


def _compositions(total: int, parts: int):
    if parts == 1:
        yield (total,)
        return
    for first in range(total + 1):
        for rest in _compositions(total - first, parts - 1):
            yield (first,) + rest


def solve_vrp_reference(
    instance: VRPInstance,
    *,
    max_customers: int = 8,
    max_evaluations: int = 2_000_000,
) -> VRPReferenceResult:
    n = len(instance.customers)
    m = len(instance.vehicles)
    if isinstance(max_customers, bool) or not isinstance(max_customers, int) or max_customers < 1:
        raise ValueError("max_customers must be a positive integer")
    if isinstance(max_evaluations, bool) or not isinstance(max_evaluations, int) or max_evaluations < 1:
        raise ValueError("max_evaluations must be a positive integer")
    if n > max_customers:
        raise VRPReferenceLimitError(f"reference VRP solver is limited to {max_customers} customers")
    search_space = factorial(n) * comb(n + m - 1, m - 1)
    if search_space > max_evaluations:
        raise VRPReferenceLimitError(f"reference VRP search-space upper bound {search_space} exceeds max_evaluations={max_evaluations}")
    customer_ids = tuple(c.id for c in instance.customers)
    best_routes: tuple[VRPRoute, ...] | None = None
    best_objective = float("inf")
    evaluations = 0
    started = perf_counter()
    for order in permutations(customer_ids):
        for sizes in _compositions(n, m):
            evaluations += 1
            routes: list[VRPRoute] = []
            pos = 0
            for vehicle, size in zip(instance.vehicles, sizes):
                route_ids = order[pos : pos + size]
                pos += size
                routes.append(VRPRoute(vehicle.id, route_ids))
            objective = sum(route_distance(instance, route) for route in routes)
            if objective >= best_objective - 1e-12:
                continue
            candidate = VRPSolution(routes, objective, method="reference-exhaustive", is_exact=True, optimality_proven=True, _proof_token=_INDEPENDENT_PROOF_TOKEN)
            report = validate_vrp_solution(instance, candidate)
            if report.valid:
                best_objective = objective
                best_routes = tuple(routes)
    if best_routes is None:
        return VRPReferenceResult(None, evaluations, search_space)
    solution = VRPSolution(
        best_routes,
        best_objective,
        method="reference-exhaustive",
        is_exact=True,
        optimality_proven=True,
        runtime_seconds=perf_counter() - started,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        metadata={"evaluations": evaluations, "search_space_upper_bound": search_space},
    )
    return VRPReferenceResult(solution, evaluations, search_space)
