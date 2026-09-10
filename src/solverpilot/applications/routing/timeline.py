from __future__ import annotations

from dataclasses import dataclass
from math import isfinite

from .errors import VRPValidationError
from .model import VRPInstance
from .solution import VRPRoute


@dataclass(frozen=True, slots=True)
class VRPVisitTimeline:
    customer_id: str
    arrival: float
    waiting: float
    service_start: float
    departure: float
    within_time_window: bool


@dataclass(frozen=True, slots=True)
class VRPRouteTimeline:
    vehicle_id: str
    used: bool
    start_time: float
    return_time: float
    route_duration: float
    visits: tuple[VRPVisitTimeline, ...]
    violations: tuple[str, ...]

    @property
    def valid(self) -> bool:
        return not self.violations


def build_route_timeline(instance: VRPInstance, route: VRPRoute, *, tolerance: float = 1e-9) -> VRPRouteTimeline:
    if isinstance(tolerance, bool):
        raise VRPValidationError("timeline tolerance must be numeric")
    tolerance = float(tolerance)
    if not isfinite(tolerance) or tolerance < 0.0:
        raise VRPValidationError("timeline tolerance must be finite and non-negative")
    if instance.travel_times is None:
        raise VRPValidationError("route timeline requires an explicit travel-time matrix")
    vehicles = instance.vehicle_map
    customers = instance.customer_map
    if route.vehicle_id not in vehicles:
        raise VRPValidationError(f"unknown vehicle {route.vehicle_id!r}")
    vehicle = vehicles[route.vehicle_id]
    current_time = vehicle.start_time
    previous_node = vehicle.start_node_id
    visits: list[VRPVisitTimeline] = []
    violations: list[str] = []
    for customer_id in route.customer_ids:
        customer = customers.get(customer_id)
        if customer is None:
            raise VRPValidationError(f"unknown customer {customer_id!r}")
        arrival = current_time + instance.travel_time(previous_node, customer.node_id)
        service_start = arrival
        if customer.time_window is not None:
            service_start = max(arrival, customer.time_window.start)
        waiting = service_start - arrival
        within = customer.time_window is None or service_start <= customer.time_window.end + tolerance
        if not within:
            violations.append(f"late_arrival:{customer.id}")
        departure = service_start + customer.service_duration
        visits.append(VRPVisitTimeline(customer.id, arrival, waiting, service_start, departure, within))
        current_time = departure
        previous_node = customer.node_id
    if route.customer_ids:
        current_time += instance.travel_time(previous_node, vehicle.end_node_id)
    # Unused vehicle is not dispatched; it incurs no route duration or depot-to-depot travel.
    used = bool(route.customer_ids)
    return_time = current_time if used else vehicle.start_time
    duration = return_time - vehicle.start_time
    if vehicle.latest_return is not None and return_time > vehicle.latest_return + tolerance:
        violations.append("vehicle_shift_late_return")
    if vehicle.max_route_duration is not None and duration > vehicle.max_route_duration + tolerance:
        violations.append("max_route_duration_exceeded")
    return VRPRouteTimeline(route.vehicle_id, used, vehicle.start_time, return_time, duration, tuple(visits), tuple(violations))
