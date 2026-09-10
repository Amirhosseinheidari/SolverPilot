from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from math import isclose, isfinite
from typing import Sequence

from .model import VRPInstance
from .solution import VRPRoute, VRPSolution
from .timeline import VRPRouteTimeline, build_route_timeline


@dataclass(frozen=True, slots=True)
class VRPValidationIssue:
    code: str
    message: str
    vehicle_id: str | None = None
    customer_id: str | None = None


@dataclass(frozen=True, slots=True)
class VRPValidationReport:
    valid: bool
    issues: tuple[VRPValidationIssue, ...]
    objective_recomputed: float | None
    objective_reported: float | None
    objective_difference: float | None
    route_loads: tuple[tuple[str, float], ...]
    route_distances: tuple[tuple[str, float], ...]
    timelines: tuple[VRPRouteTimeline, ...]
    validated_dimensions: tuple[str, ...]

    @property
    def errors(self) -> tuple[str, ...]:
        return tuple(i.message for i in self.issues)


def route_distance(instance: VRPInstance, route: VRPRoute) -> float:
    if route.vehicle_id not in instance.vehicle_map:
        from .errors import VRPValidationError
        raise VRPValidationError(f"unknown vehicle {route.vehicle_id!r}")
    unknown = [cid for cid in route.customer_ids if cid not in instance.customer_map]
    if unknown:
        from .errors import VRPValidationError
        raise VRPValidationError(f"unknown customer(s) in route: {sorted(set(unknown))}")
    if not route.customer_ids:
        return 0.0
    vehicles = instance.vehicle_map
    customers = instance.customer_map
    vehicle = vehicles[route.vehicle_id]
    total = 0.0
    previous = vehicle.start_node_id
    for cid in route.customer_ids:
        c = customers[cid]
        total += instance.distance(previous, c.node_id)
        previous = c.node_id
    total += instance.distance(previous, vehicle.end_node_id)
    return float(total)


def _route_load(instance: VRPInstance, route: VRPRoute) -> float:
    customers = instance.customer_map
    return float(sum(customers[cid].demand for cid in route.customer_ids if cid in customers))


def validate_vrp_solution(
    instance: VRPInstance,
    solution_or_routes: VRPSolution | Sequence[VRPRoute],
    *,
    reported_objective: float | None = None,
    atol: float = 1e-8,
    rtol: float = 1e-8,
) -> VRPValidationReport:
    if isinstance(atol, bool) or isinstance(rtol, bool):
        raise ValueError("VRP validation tolerances must be numeric")
    atol = float(atol); rtol = float(rtol)
    if not isfinite(atol) or not isfinite(rtol) or atol < 0.0 or rtol < 0.0:
        raise ValueError("VRP validation tolerances must be finite and non-negative")
    if isinstance(solution_or_routes, VRPSolution):
        solution = solution_or_routes
        routes = solution.routes
        if reported_objective is None:
            reported_objective = solution.objective
        declared_unassigned = set(solution.unassigned_customer_ids)
    else:
        solution = None
        routes = tuple(solution_or_routes)
        declared_unassigned = set()
    issues: list[VRPValidationIssue] = []
    vehicle_map = instance.vehicle_map
    customer_map = instance.customer_map
    seen_vehicles: Counter[str] = Counter()
    served: list[str] = []
    route_loads: list[tuple[str, float]] = []
    route_distances: list[tuple[str, float]] = []
    timelines: list[VRPRouteTimeline] = []
    objective = 0.0
    for route in routes:
        if not isinstance(route, VRPRoute):
            issues.append(VRPValidationIssue("invalid_route_type", "solution routes must contain VRPRoute objects"))
            continue
        seen_vehicles[route.vehicle_id] += 1
        vehicle = vehicle_map.get(route.vehicle_id)
        if vehicle is None:
            issues.append(VRPValidationIssue("unknown_vehicle", f"route references unknown vehicle {route.vehicle_id!r}", route.vehicle_id))
            continue
        if seen_vehicles[route.vehicle_id] > 1:
            issues.append(VRPValidationIssue("duplicate_vehicle_route", f"vehicle {route.vehicle_id!r} has multiple routes", route.vehicle_id))
        local = Counter(route.customer_ids)
        for cid, count in local.items():
            if cid not in customer_map:
                issues.append(VRPValidationIssue("unknown_customer", f"route references unknown customer {cid!r}", route.vehicle_id, cid))
            elif count > 1:
                issues.append(VRPValidationIssue("duplicate_customer_in_route", f"customer {cid!r} is repeated in one route", route.vehicle_id, cid))
        if any(cid not in customer_map for cid in route.customer_ids):
            continue
        served.extend(route.customer_ids)
        load = _route_load(instance, route)
        route_loads.append((route.vehicle_id, load))
        if load > vehicle.capacity + atol:
            issues.append(VRPValidationIssue("capacity_exceeded", f"vehicle {route.vehicle_id!r} load {load} exceeds capacity {vehicle.capacity}", route.vehicle_id))
        dist = route_distance(instance, route)
        route_distances.append((route.vehicle_id, dist))
        objective += dist
        if instance.has_time_constraints:
            timeline = build_route_timeline(instance, route, tolerance=atol)
            timelines.append(timeline)
            for code in timeline.violations:
                customer_id = code.split(":", 1)[1] if code.startswith("late_arrival:") else None
                base_code = "time_window_violation" if customer_id is not None else code
                issues.append(VRPValidationIssue(base_code, f"route {route.vehicle_id!r} violates {code}", route.vehicle_id, customer_id))
    served_counts = Counter(served)
    duplicated_global = sorted(cid for cid, count in served_counts.items() if count > 1)
    for cid in duplicated_global:
        issues.append(VRPValidationIssue("duplicate_customer_assignment", f"customer {cid!r} is assigned more than once", customer_id=cid))
    required = set(customer_map)
    served_set = set(served_counts)
    missing = sorted(required - served_set)
    for cid in missing:
        issues.append(VRPValidationIssue("unassigned_customer", f"customer {cid!r} is not assigned", customer_id=cid))
    unknown_unassigned = sorted(declared_unassigned - required)
    for cid in unknown_unassigned:
        issues.append(VRPValidationIssue("unknown_unassigned_customer", f"unassigned list contains unknown customer {cid!r}", customer_id=cid))
    contradictory = sorted(declared_unassigned & served_set)
    for cid in contradictory:
        issues.append(VRPValidationIssue("served_and_unassigned", f"customer {cid!r} is both served and declared unassigned", customer_id=cid))
    if solution is not None and set(missing) != declared_unassigned:
        issues.append(VRPValidationIssue("unassigned_declaration_mismatch", "declared unassigned customers do not exactly match missing coverage"))
    reported: float | None = None
    difference: float | None = None
    if reported_objective is not None:
        if isinstance(reported_objective, bool):
            issues.append(VRPValidationIssue("invalid_objective", "reported objective must be numeric, not bool"))
        else:
            try:
                reported = float(reported_objective)
            except (TypeError, ValueError):
                issues.append(VRPValidationIssue("invalid_objective", "reported objective must be numeric"))
            else:
                if not isfinite(reported):
                    issues.append(VRPValidationIssue("invalid_objective", "reported objective must be finite"))
                else:
                    difference = reported - objective
                    if not isclose(reported, objective, abs_tol=atol, rel_tol=rtol):
                        issues.append(VRPValidationIssue("objective_mismatch", "reported objective does not match independently recomputed route distance"))
    dims = ["coverage", "vehicle_references", "capacity", "distance", "objective"]
    if instance.has_time_constraints:
        dims.extend(("travel_time", "customer_time_windows", "vehicle_shift", "route_duration"))
    return VRPValidationReport(not issues, tuple(issues), objective, reported, difference, tuple(route_loads), tuple(route_distances), tuple(timelines), tuple(dims))
