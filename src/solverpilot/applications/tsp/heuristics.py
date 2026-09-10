from __future__ import annotations

from time import perf_counter
from typing import Sequence

from .distance import analyze_distance_matrix, route_cost
from .errors import TSPValidationError
from .model import TSPInstance
from .solution import TSPSolution
from .validation import validate_tsp_route


def _start(instance: TSPInstance, value: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or not 0 <= value < instance.n_nodes:
        raise TSPValidationError("start_index must be an in-range integer")
    return value


def solve_tsp_nearest_neighbor(instance: TSPInstance, *, start_index: int = 0) -> TSPSolution:
    start = _start(instance, start_index)
    t0 = perf_counter()
    unvisited = set(range(instance.n_nodes))
    unvisited.remove(start)
    route = [start]
    current = start
    while unvisited:
        nxt = min(unvisited, key=lambda j: (instance.distances.values[current][j], j))
        route.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    route.append(start)
    objective = route_cost(instance, route)
    return TSPSolution(route, objective, method="nearest-neighbor", is_exact=False, optimality_proven=False, runtime_seconds=perf_counter()-t0, metadata={"start_index": start})


def solve_tsp_multistart_nearest_neighbor(instance: TSPInstance, *, start_indices: Sequence[int] | None = None) -> TSPSolution:
    if start_indices is None:
        starts = tuple(range(instance.n_nodes))
    else:
        if isinstance(start_indices, (str, bytes)):
            raise TSPValidationError("start_indices must be a sequence of integers")
        starts = tuple(dict.fromkeys(_start(instance, x) for x in start_indices))
        if not starts:
            raise TSPValidationError("start_indices must not be empty")
    t0 = perf_counter()
    candidates = [solve_tsp_nearest_neighbor(instance, start_index=s) for s in starts]
    best = min(candidates, key=lambda x: (x.objective, x.route))
    return TSPSolution(best.route, best.objective, method="multistart-nearest-neighbor", is_exact=False, optimality_proven=False, runtime_seconds=perf_counter()-t0, metadata={"start_indices": starts, "candidate_objectives": tuple(x.objective for x in candidates)})


def solve_tsp_two_opt(
    instance: TSPInstance,
    *,
    initial_route: Sequence[int] | TSPSolution | None = None,
    start_index: int = 0,
    max_iterations: int = 1000,
    strategy: str = "best",
    symmetry_atol: float = 1e-12,
) -> TSPSolution:
    if isinstance(max_iterations, bool) or not isinstance(max_iterations, int) or max_iterations < 1:
        raise TSPValidationError("max_iterations must be a positive integer")
    strategy = str(strategy).strip().lower()
    if strategy not in {"best", "first"}:
        raise TSPValidationError("strategy must be 'best' or 'first'")
    start = _start(instance, start_index)
    if initial_route is None:
        route = (start, *(i for i in range(instance.n_nodes) if i != start), start)
    elif isinstance(initial_route, TSPSolution):
        route = tuple(initial_route.route)
    else:
        route = tuple(initial_route)
    report = validate_tsp_route(instance, route)
    if not report.valid:
        raise TSPValidationError("initial_route is invalid: " + "; ".join(report.errors))
    if route[0] != start:
        raise TSPValidationError("initial_route must start at start_index")
    t0 = perf_counter()
    symmetric = analyze_distance_matrix(instance.distances, atol=symmetry_atol, check_triangle=False).symmetric
    current = route
    current_cost = route_cost(instance, current)
    initial_cost = current_cost
    moves_evaluated = 0
    improvements = 0
    iterations = 0
    n = instance.n_nodes
    for iteration in range(max_iterations):
        iterations = iteration + 1
        best_route: tuple[int, ...] | None = None
        best_cost = current_cost
        for i in range(1, n - 1):
            for k in range(i + 1, n):
                moves_evaluated += 1
                candidate = current[:i] + tuple(reversed(current[i:k+1])) + current[k+1:]
                if symmetric:
                    a, b, c, d = current[i-1], current[i], current[k], current[k+1]
                    candidate_cost = current_cost - instance.distances.values[a][b] - instance.distances.values[c][d] + instance.distances.values[a][c] + instance.distances.values[b][d]
                else:
                    # Reversal changes all internal arc directions for an asymmetric TSP.
                    candidate_cost = route_cost(instance, candidate)
                if candidate_cost + 1e-12 < best_cost:
                    best_route, best_cost = candidate, float(candidate_cost)
                    if strategy == "first":
                        break
            if strategy == "first" and best_route is not None:
                break
        if best_route is None:
            break
        current = best_route
        current_cost = route_cost(instance, current)  # keep accumulated deltas from becoming the authority
        improvements += 1
    final_cost = route_cost(instance, current)
    return TSPSolution(current, final_cost, method="two-opt", is_exact=False, optimality_proven=False, runtime_seconds=perf_counter()-t0, metadata={"start_index": start, "symmetric_fast_delta": symmetric, "initial_cost": initial_cost, "improvements": improvements, "moves_evaluated": moves_evaluated, "iterations": iterations, "strategy": strategy})
