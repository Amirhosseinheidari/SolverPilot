from __future__ import annotations

from dataclasses import dataclass, field
from math import hypot, isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import TSPValidationError


def _frozen_metadata(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


@dataclass(frozen=True, slots=True)
class TSPNode:
    id: str
    x: float | None = None
    y: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        node_id = str(self.id).strip()
        if not node_id:
            raise TSPValidationError("TSP node id must be a non-empty string")
        object.__setattr__(self, "id", node_id)
        if (self.x is None) != (self.y is None):
            raise TSPValidationError("TSP coordinates must provide both x and y or neither")
        if self.x is not None:
            if isinstance(self.x, bool) or isinstance(self.y, bool):
                raise TSPValidationError("TSP node coordinates must be numeric, not bool")
            try:
                x = float(self.x)
                y = float(self.y)
            except (TypeError, ValueError) as exc:
                raise TSPValidationError("TSP node coordinates must be numeric") from exc
            if not isfinite(x) or not isfinite(y):
                raise TSPValidationError("TSP node coordinates must be finite")
            object.__setattr__(self, "x", x)
            object.__setattr__(self, "y", y)
        object.__setattr__(self, "metadata", _frozen_metadata(self.metadata))


@dataclass(frozen=True, slots=True)
class TSPDistanceMatrix:
    values: tuple[tuple[float, ...], ...]

    def __init__(self, values: Sequence[Sequence[float]]):
        try:
            raw_rows = tuple(tuple(row) for row in values)
        except TypeError as exc:
            raise TSPValidationError("TSP distance matrix must be a sequence of rows") from exc
        rows_list: list[tuple[float, ...]] = []
        for i, row in enumerate(raw_rows):
            normalized: list[float] = []
            for j, value in enumerate(row):
                if isinstance(value, bool):
                    raise TSPValidationError(f"distance at ({i}, {j}) must be numeric, not bool")
                try:
                    normalized.append(float(value))
                except (TypeError, ValueError) as exc:
                    raise TSPValidationError(f"distance at ({i}, {j}) must be numeric") from exc
            rows_list.append(tuple(normalized))
        rows = tuple(rows_list)
        if len(rows) < 2:
            raise TSPValidationError("a TSP distance matrix must contain at least two nodes")
        n = len(rows)
        if any(len(row) != n for row in rows):
            raise TSPValidationError("TSP distance matrix must be square")
        for i, row in enumerate(rows):
            for j, value in enumerate(row):
                if not isfinite(value):
                    raise TSPValidationError(f"distance at ({i}, {j}) must be finite")
                if value < 0.0:
                    raise TSPValidationError(f"distance at ({i}, {j}) must be non-negative")
        object.__setattr__(self, "values", rows)

    @property
    def n_nodes(self) -> int:
        return len(self.values)

    def edge_cost(self, start: int, end: int) -> float:
        if isinstance(start, bool) or isinstance(end, bool):
            raise TSPValidationError("node indices must be integers, not bool")
        if not isinstance(start, int) or not isinstance(end, int):
            raise TSPValidationError("node indices must be integers")
        if not (0 <= start < self.n_nodes and 0 <= end < self.n_nodes):
            raise TSPValidationError("node index outside TSP distance matrix")
        return self.values[start][end]


@dataclass(frozen=True, slots=True)
class TSPInstance:
    nodes: tuple[TSPNode, ...]
    distances: TSPDistanceMatrix
    name: str = "tsp"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        nodes: Sequence[TSPNode],
        distances: TSPDistanceMatrix | Sequence[Sequence[float]],
        *,
        name: str = "tsp",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        node_tuple = tuple(nodes)
        matrix = distances if isinstance(distances, TSPDistanceMatrix) else TSPDistanceMatrix(distances)
        if len(node_tuple) != matrix.n_nodes:
            raise TSPValidationError("node count must match the distance matrix size")
        ids = tuple(node.id for node in node_tuple)
        if len(set(ids)) != len(ids):
            raise TSPValidationError("TSP node ids must be unique")
        clean_name = str(name).strip()
        if not clean_name:
            raise TSPValidationError("TSP instance name must be non-empty")
        object.__setattr__(self, "nodes", node_tuple)
        object.__setattr__(self, "distances", matrix)
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "metadata", _frozen_metadata(metadata))

    @property
    def n_nodes(self) -> int:
        return len(self.nodes)

    @property
    def node_ids(self) -> tuple[str, ...]:
        return tuple(node.id for node in self.nodes)

    @classmethod
    def from_distance_matrix(
        cls,
        matrix: Sequence[Sequence[float]],
        *,
        node_ids: Sequence[str] | None = None,
        name: str = "tsp",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TSPInstance":
        distances = TSPDistanceMatrix(matrix)
        ids = tuple(str(i) for i in range(distances.n_nodes)) if node_ids is None else tuple(node_ids)
        if len(ids) != distances.n_nodes:
            raise TSPValidationError("node_ids length must match the distance matrix")
        return cls(tuple(TSPNode(node_id) for node_id in ids), distances, name=name, metadata=metadata)

    @classmethod
    def from_coordinates(
        cls,
        coordinates: Sequence[tuple[float, float]],
        *,
        node_ids: Sequence[str] | None = None,
        round_digits: int | None = None,
        name: str = "tsp",
        metadata: Mapping[str, Any] | None = None,
    ) -> "TSPInstance":
        if len(coordinates) < 2:
            raise TSPValidationError("a TSP instance requires at least two coordinates")
        if round_digits is not None:
            if isinstance(round_digits, bool) or not isinstance(round_digits, int) or round_digits < 0:
                raise TSPValidationError("round_digits must be a non-negative integer or None")
        ids = tuple(str(i) for i in range(len(coordinates))) if node_ids is None else tuple(node_ids)
        if len(ids) != len(coordinates):
            raise TSPValidationError("node_ids length must match coordinates")
        nodes = tuple(TSPNode(node_id, x, y) for node_id, (x, y) in zip(ids, coordinates))
        rows: list[tuple[float, ...]] = []
        for a in nodes:
            if a.x is None or a.y is None:
                raise TSPValidationError("coordinate-backed TSP node unexpectedly lacks coordinates")
            row: list[float] = []
            for b in nodes:
                if b.x is None or b.y is None:
                    raise TSPValidationError("coordinate-backed TSP node unexpectedly lacks coordinates")
                value = hypot(a.x - b.x, a.y - b.y)
                row.append(round(value, round_digits) if round_digits is not None else value)
            rows.append(tuple(row))
        return cls(nodes, rows, name=name, metadata=metadata)
