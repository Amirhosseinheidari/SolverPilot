from __future__ import annotations

from dataclasses import dataclass, field
from math import isfinite
from types import MappingProxyType
from typing import Any, Mapping, Sequence

from .errors import VRPValidationError


def _meta(value: Mapping[str, Any] | None) -> Mapping[str, Any]:
    return MappingProxyType(dict(value or {}))


def _clean_id(value: object, label: str) -> str:
    text = str(value).strip()
    if not text:
        raise VRPValidationError(f"{label} must be a non-empty string")
    return text


def _number(value: object, label: str, *, positive: bool = False) -> float:
    if isinstance(value, bool):
        raise VRPValidationError(f"{label} must be numeric, not bool")
    try:
        out = float(value)
    except (TypeError, ValueError) as exc:
        raise VRPValidationError(f"{label} must be numeric") from exc
    if not isfinite(out):
        raise VRPValidationError(f"{label} must be finite")
    if positive and out <= 0.0:
        raise VRPValidationError(f"{label} must be positive")
    if not positive and out < 0.0:
        raise VRPValidationError(f"{label} must be non-negative")
    return out


@dataclass(frozen=True, slots=True)
class VRPTimeWindow:
    start: float
    end: float

    def __post_init__(self) -> None:
        start = _number(self.start, "time-window start")
        end = _number(self.end, "time-window end")
        if end < start:
            raise VRPValidationError("time-window end must be >= start")
        object.__setattr__(self, "start", start)
        object.__setattr__(self, "end", end)


@dataclass(frozen=True, slots=True)
class VRPNode:
    id: str
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_id(self.id, "node id"))
        object.__setattr__(self, "metadata", _meta(self.metadata))


@dataclass(frozen=True, slots=True)
class VRPCustomer:
    id: str
    node_id: str
    demand: float = 0.0
    service_duration: float = 0.0
    time_window: VRPTimeWindow | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        object.__setattr__(self, "id", _clean_id(self.id, "customer id"))
        object.__setattr__(self, "node_id", _clean_id(self.node_id, "customer node_id"))
        object.__setattr__(self, "demand", _number(self.demand, f"customer {self.id} demand"))
        object.__setattr__(self, "service_duration", _number(self.service_duration, f"customer {self.id} service_duration"))
        if self.time_window is not None and not isinstance(self.time_window, VRPTimeWindow):
            raise VRPValidationError("customer time_window must be VRPTimeWindow or None")
        object.__setattr__(self, "metadata", _meta(self.metadata))


@dataclass(frozen=True, slots=True)
class VRPVehicle:
    id: str
    start_node_id: str
    end_node_id: str | None = None
    capacity: float = 1.0
    shift_window: VRPTimeWindow | None = None
    max_route_duration: float | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __post_init__(self) -> None:
        vehicle_id = _clean_id(self.id, "vehicle id")
        start = _clean_id(self.start_node_id, f"vehicle {vehicle_id} start_node_id")
        end = start if self.end_node_id is None else _clean_id(self.end_node_id, f"vehicle {vehicle_id} end_node_id")
        capacity = _number(self.capacity, f"vehicle {vehicle_id} capacity", positive=True)
        if self.shift_window is not None and not isinstance(self.shift_window, VRPTimeWindow):
            raise VRPValidationError("vehicle shift_window must be VRPTimeWindow or None")
        duration = None
        if self.max_route_duration is not None:
            duration = _number(self.max_route_duration, f"vehicle {vehicle_id} max_route_duration", positive=True)
        object.__setattr__(self, "id", vehicle_id)
        object.__setattr__(self, "start_node_id", start)
        object.__setattr__(self, "end_node_id", end)
        object.__setattr__(self, "capacity", capacity)
        object.__setattr__(self, "max_route_duration", duration)
        object.__setattr__(self, "metadata", _meta(self.metadata))

    @property
    def start_time(self) -> float:
        return 0.0 if self.shift_window is None else self.shift_window.start

    @property
    def latest_return(self) -> float | None:
        return None if self.shift_window is None else self.shift_window.end


@dataclass(frozen=True, slots=True)
class VRPMatrix:
    values: tuple[tuple[float, ...], ...]

    def __init__(self, values: Sequence[Sequence[float]], *, label: str = "matrix") -> None:
        try:
            raw = tuple(tuple(row) for row in values)
        except TypeError as exc:
            raise VRPValidationError(f"{label} must be a sequence of rows") from exc
        if len(raw) < 2:
            raise VRPValidationError(f"{label} must contain at least two nodes")
        n = len(raw)
        if any(len(row) != n for row in raw):
            raise VRPValidationError(f"{label} must be square")
        rows: list[tuple[float, ...]] = []
        for i, row in enumerate(raw):
            out: list[float] = []
            for j, value in enumerate(row):
                if isinstance(value, bool):
                    raise VRPValidationError(f"{label}[{i},{j}] must be numeric, not bool")
                try:
                    v = float(value)
                except (TypeError, ValueError) as exc:
                    raise VRPValidationError(f"{label}[{i},{j}] must be numeric") from exc
                if not isfinite(v) or v < 0.0:
                    raise VRPValidationError(f"{label}[{i},{j}] must be finite and non-negative")
                out.append(v)
            rows.append(tuple(out))
        object.__setattr__(self, "values", tuple(rows))

    @property
    def n_nodes(self) -> int:
        return len(self.values)

    def edge(self, i: int, j: int) -> float:
        if isinstance(i, bool) or isinstance(j, bool) or not isinstance(i, int) or not isinstance(j, int):
            raise VRPValidationError("matrix indices must be integers")
        if not 0 <= i < self.n_nodes or not 0 <= j < self.n_nodes:
            raise VRPValidationError("matrix index out of range")
        return self.values[i][j]


@dataclass(frozen=True, slots=True)
class VRPInstance:
    nodes: tuple[VRPNode, ...]
    customers: tuple[VRPCustomer, ...]
    vehicles: tuple[VRPVehicle, ...]
    distances: VRPMatrix
    travel_times: VRPMatrix | None = None
    name: str = "vrp"
    metadata: Mapping[str, Any] = field(default_factory=dict, compare=False)

    def __init__(
        self,
        nodes: Sequence[VRPNode],
        customers: Sequence[VRPCustomer],
        vehicles: Sequence[VRPVehicle],
        distances: VRPMatrix | Sequence[Sequence[float]],
        *,
        travel_times: VRPMatrix | Sequence[Sequence[float]] | None = None,
        name: str = "vrp",
        metadata: Mapping[str, Any] | None = None,
    ) -> None:
        node_tuple = tuple(nodes)
        customer_tuple = tuple(customers)
        vehicle_tuple = tuple(vehicles)
        if len(node_tuple) < 2:
            raise VRPValidationError("VRP requires at least two nodes")
        if not customer_tuple:
            raise VRPValidationError("VRP requires at least one customer")
        if not vehicle_tuple:
            raise VRPValidationError("VRP requires at least one vehicle")
        node_ids = tuple(n.id for n in node_tuple)
        customer_ids = tuple(c.id for c in customer_tuple)
        vehicle_ids = tuple(v.id for v in vehicle_tuple)
        if len(set(node_ids)) != len(node_ids):
            raise VRPValidationError("VRP node ids must be unique")
        if len(set(customer_ids)) != len(customer_ids):
            raise VRPValidationError("VRP customer ids must be unique")
        if len(set(vehicle_ids)) != len(vehicle_ids):
            raise VRPValidationError("VRP vehicle ids must be unique")
        node_set = set(node_ids)
        unknown_customer_nodes = sorted({c.node_id for c in customer_tuple} - node_set)
        unknown_vehicle_nodes = sorted({x for v in vehicle_tuple for x in (v.start_node_id, v.end_node_id)} - node_set)
        if unknown_customer_nodes:
            raise VRPValidationError(f"customers reference unknown nodes: {unknown_customer_nodes}")
        if unknown_vehicle_nodes:
            raise VRPValidationError(f"vehicles reference unknown nodes: {unknown_vehicle_nodes}")
        d = distances if isinstance(distances, VRPMatrix) else VRPMatrix(distances, label="distance matrix")
        if d.n_nodes != len(node_tuple):
            raise VRPValidationError("distance matrix size must match node count")
        tt = None
        if travel_times is not None:
            tt = travel_times if isinstance(travel_times, VRPMatrix) else VRPMatrix(travel_times, label="travel-time matrix")
            if tt.n_nodes != len(node_tuple):
                raise VRPValidationError("travel-time matrix size must match node count")
        requires_time = any(c.time_window is not None for c in customer_tuple) or any(
            v.shift_window is not None or v.max_route_duration is not None for v in vehicle_tuple
        )
        if requires_time and tt is None:
            raise VRPValidationError("time-constrained VRP requires an explicit travel-time matrix; distance is never used as implicit time")
        clean_name = _clean_id(name, "VRP name")
        object.__setattr__(self, "nodes", node_tuple)
        object.__setattr__(self, "customers", customer_tuple)
        object.__setattr__(self, "vehicles", vehicle_tuple)
        object.__setattr__(self, "distances", d)
        object.__setattr__(self, "travel_times", tt)
        object.__setattr__(self, "name", clean_name)
        object.__setattr__(self, "metadata", _meta(metadata))

    @property
    def node_index(self) -> Mapping[str, int]:
        return MappingProxyType({node.id: i for i, node in enumerate(self.nodes)})

    @property
    def customer_map(self) -> Mapping[str, VRPCustomer]:
        return MappingProxyType({c.id: c for c in self.customers})

    @property
    def vehicle_map(self) -> Mapping[str, VRPVehicle]:
        return MappingProxyType({v.id: v for v in self.vehicles})

    @property
    def has_time_constraints(self) -> bool:
        return any(c.time_window is not None for c in self.customers) or any(
            v.shift_window is not None or v.max_route_duration is not None for v in self.vehicles
        )

    def distance(self, from_node_id: str, to_node_id: str) -> float:
        idx = self.node_index
        try:
            return self.distances.values[idx[from_node_id]][idx[to_node_id]]
        except KeyError as exc:
            raise VRPValidationError(f"unknown node in distance lookup: {exc.args[0]}") from exc

    def travel_time(self, from_node_id: str, to_node_id: str) -> float:
        if self.travel_times is None:
            raise VRPValidationError("travel-time matrix is unavailable")
        idx = self.node_index
        try:
            return self.travel_times.values[idx[from_node_id]][idx[to_node_id]]
        except KeyError as exc:
            raise VRPValidationError(f"unknown node in travel-time lookup: {exc.args[0]}") from exc
