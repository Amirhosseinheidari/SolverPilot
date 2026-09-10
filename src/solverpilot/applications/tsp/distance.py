from __future__ import annotations

from dataclasses import dataclass
from math import isclose
from typing import Sequence

from .errors import TSPValidationError
from .model import TSPDistanceMatrix, TSPInstance


@dataclass(frozen=True, slots=True)
class TSPDistanceAnalysis:
    symmetric: bool
    zero_diagonal: bool
    triangle_inequality: bool | None
    metric: bool | None
    atol: float


def analyze_distance_matrix(
    matrix: TSPDistanceMatrix | Sequence[Sequence[float]],
    *,
    atol: float = 1e-12,
    check_triangle: bool = True,
) -> TSPDistanceAnalysis:
    if not isinstance(atol, (int, float)) or isinstance(atol, bool) or float(atol) < 0.0:
        raise TSPValidationError("atol must be a non-negative finite number")
    atol = float(atol)
    if atol == float("inf") or atol != atol:
        raise TSPValidationError("atol must be finite")
    dm = matrix if isinstance(matrix, TSPDistanceMatrix) else TSPDistanceMatrix(matrix)
    n = dm.n_nodes
    symmetric = all(isclose(dm.values[i][j], dm.values[j][i], rel_tol=0.0, abs_tol=atol) for i in range(n) for j in range(i + 1, n))
    zero_diagonal = all(abs(dm.values[i][i]) <= atol for i in range(n))
    triangle: bool | None = None
    metric: bool | None = None
    if check_triangle:
        triangle = True
        for i in range(n):
            for j in range(n):
                dij = dm.values[i][j]
                for k in range(n):
                    if dij > dm.values[i][k] + dm.values[k][j] + atol:
                        triangle = False
                        break
                if not triangle:
                    break
            if not triangle:
                break
        metric = symmetric and zero_diagonal and triangle
    return TSPDistanceAnalysis(symmetric, zero_diagonal, triangle, metric, atol)


def route_cost(instance: TSPInstance, route: Sequence[int]) -> float:
    if len(route) < 2:
        raise TSPValidationError("route must contain at least two entries")
    total = 0.0
    for a, b in zip(route, route[1:]):
        total += instance.distances.edge_cost(a, b)
    return float(total)
