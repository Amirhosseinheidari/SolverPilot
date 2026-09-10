from __future__ import annotations

from itertools import permutations
from math import inf
from time import perf_counter

from .distance import route_cost
from .errors import TSPReferenceLimitError, TSPValidationError
from .model import TSPInstance
from .solution import TSPSolution, _INDEPENDENT_PROOF_TOKEN


def _validate_start(instance: TSPInstance, start_index: int) -> int:
    if isinstance(start_index, bool) or not isinstance(start_index, int):
        raise TSPValidationError("start_index must be an integer")
    if not 0 <= start_index < instance.n_nodes:
        raise TSPValidationError("start_index is outside the TSP instance")
    return start_index


def _validate_limit(max_nodes: int, *, minimum: int = 2) -> int:
    if isinstance(max_nodes, bool) or not isinstance(max_nodes, int) or max_nodes < minimum:
        raise TSPValidationError(f"max_nodes must be an integer >= {minimum}")
    return max_nodes


def solve_tsp_brute_force(instance: TSPInstance, *, start_index: int = 0, max_nodes: int = 10) -> TSPSolution:
    start = _validate_start(instance, start_index)
    limit = _validate_limit(max_nodes)
    if instance.n_nodes > limit:
        raise TSPReferenceLimitError(f"brute force refused {instance.n_nodes} nodes because max_nodes={limit}")
    t0 = perf_counter()
    rest = tuple(i for i in range(instance.n_nodes) if i != start)
    best_route: tuple[int, ...] | None = None
    best_cost = inf
    examined = 0
    for ordering in permutations(rest):
        route = (start, *ordering, start)
        cost = route_cost(instance, route)
        examined += 1
        if cost < best_cost or (cost == best_cost and (best_route is None or route < best_route)):
            best_route, best_cost = route, cost
    if best_route is None:
        raise RuntimeError("TSP brute-force search produced no route")
    return TSPSolution(
        best_route,
        best_cost,
        method="brute-force",
        is_exact=True,
        optimality_proven=True,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        runtime_seconds=perf_counter() - t0,
        metadata={"start_index": start, "permutations_evaluated": examined, "max_nodes": limit},
    )


def solve_tsp_held_karp(instance: TSPInstance, *, start_index: int = 0, max_nodes: int = 18) -> TSPSolution:
    start = _validate_start(instance, start_index)
    limit = _validate_limit(max_nodes)
    if instance.n_nodes > limit:
        raise TSPReferenceLimitError(f"Held-Karp refused {instance.n_nodes} nodes because max_nodes={limit}")
    t0 = perf_counter()
    nodes = tuple(i for i in range(instance.n_nodes) if i != start)
    bit = {node: pos for pos, node in enumerate(nodes)}
    full = (1 << len(nodes)) - 1
    cost: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int | None] = {}
    for node in nodes:
        mask = 1 << bit[node]
        cost[(mask, node)] = instance.distances.values[start][node]
        parent[(mask, node)] = None
    for mask in range(1, full + 1):
        if mask.bit_count() <= 1:
            continue
        for endpoint in nodes:
            ebit = 1 << bit[endpoint]
            if not (mask & ebit):
                continue
            prev_mask = mask ^ ebit
            best = inf
            best_prev: int | None = None
            for prev in nodes:
                pbit = 1 << bit[prev]
                if not (prev_mask & pbit):
                    continue
                candidate = cost[(prev_mask, prev)] + instance.distances.values[prev][endpoint]
                if candidate < best or (candidate == best and (best_prev is None or prev < best_prev)):
                    best = candidate
                    best_prev = prev
            cost[(mask, endpoint)] = best
            parent[(mask, endpoint)] = best_prev
    best_total = inf
    end: int | None = None
    for endpoint in nodes:
        candidate = cost[(full, endpoint)] + instance.distances.values[endpoint][start]
        if candidate < best_total or (candidate == best_total and (end is None or endpoint < end)):
            best_total = candidate
            end = endpoint
    if end is None:
        raise RuntimeError("Held-Karp search produced no terminal node")
    rev: list[int] = []
    mask = full
    cur: int | None = end
    while cur is not None:
        rev.append(cur)
        prev = parent[(mask, cur)]
        mask ^= 1 << bit[cur]
        cur = prev
    route = (start, *reversed(rev), start)
    # Recompute independently from the reconstructed route; the DP value is not trusted as output cost.
    objective = route_cost(instance, route)
    return TSPSolution(
        route,
        objective,
        method="held-karp",
        is_exact=True,
        optimality_proven=True,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        runtime_seconds=perf_counter() - t0,
        metadata={"start_index": start, "states_stored": len(cost), "max_nodes": limit, "dp_objective": best_total},
    )
