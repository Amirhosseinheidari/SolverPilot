from __future__ import annotations

from itertools import permutations
from fractions import Fraction
from time import perf_counter

from .distance import route_cost
from .errors import TSPReferenceLimitError, TSPValidationError
from .model import TSPInstance
from .solution import TSPSolution, _INDEPENDENT_PROOF_TOKEN


def _integer_costs(instance):
    """A common dyadic denominator preserves exact ordering of binary64 costs."""
    ratios = [[float(v).as_integer_ratio() for v in row] for row in instance.distances.values]
    denominator = max(d for row in ratios for _, d in row)
    return [[n * (denominator // d) for n, d in row] for row in ratios], denominator


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
    costs, denominator = _integer_costs(instance)
    best_cost = None
    examined = 0
    for ordering in permutations(rest):
        route = (start, *ordering, start)
        cost = sum(costs[a][b] for a, b in zip(route, route[1:]))
        examined += 1
        if best_cost is None or cost < best_cost or (cost == best_cost and (best_route is None or route < best_route)):
            best_route, best_cost = route, cost
    if best_route is None:
        raise RuntimeError("TSP brute-force search produced no route")
    return TSPSolution(
        best_route,
        route_cost(instance, best_route),
        method="brute-force",
        is_exact=True,
        optimality_proven=True,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        runtime_seconds=perf_counter() - t0,
        metadata={"start_index": start, "permutations_evaluated": examined, "max_nodes": limit, "comparison_arithmetic": "exact-dyadic", "exact_objective": str(Fraction(best_cost, denominator))},
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
    costs, denominator = _integer_costs(instance)
    cost: dict[tuple[int, int], int] = {}
    parent: dict[tuple[int, int], int | None] = {}
    for node in nodes:
        mask = 1 << bit[node]
        cost[(mask, node)] = costs[start][node]
        parent[(mask, node)] = None
    for mask in range(1, full + 1):
        if mask.bit_count() <= 1:
            continue
        for endpoint in nodes:
            ebit = 1 << bit[endpoint]
            if not (mask & ebit):
                continue
            prev_mask = mask ^ ebit
            best = None
            best_prev: int | None = None
            for prev in nodes:
                pbit = 1 << bit[prev]
                if not (prev_mask & pbit):
                    continue
                candidate = cost[(prev_mask, prev)] + costs[prev][endpoint]
                if best is None or candidate < best or (candidate == best and (best_prev is None or prev < best_prev)):
                    best = candidate
                    best_prev = prev
            cost[(mask, endpoint)] = best
            parent[(mask, endpoint)] = best_prev
    best_total = None
    end: int | None = None
    for endpoint in nodes:
        candidate = cost[(full, endpoint)] + costs[endpoint][start]
        if best_total is None or candidate < best_total or (candidate == best_total and (end is None or endpoint < end)):
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
        metadata={"start_index": start, "states_stored": len(cost), "max_nodes": limit, "dp_objective": float(Fraction(best_total, denominator)), "comparison_arithmetic": "exact-dyadic", "exact_objective": str(Fraction(best_total, denominator))},
    )
