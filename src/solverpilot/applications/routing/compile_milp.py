from __future__ import annotations

from dataclasses import dataclass
from math import isfinite
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import sparse

from solverpilot.backends import ScipyHighsBackend
from solverpilot.problem import LinearProblem, ObjectiveSense, VariableDomain
from solverpilot.runtime import SolveResult, execute
from solverpilot.validate import CandidateSolution, validate_solution

from .errors import VRPCompileError
from .model import VRPInstance
from .solution import VRPRoute, VRPSolution, _INDEPENDENT_PROOF_TOKEN
from .validation import VRPValidationReport, route_distance, validate_vrp_solution

SOURCE = "__source__"
SINK = "__sink__"


@dataclass(frozen=True, slots=True)
class VRPMILPCompilation:
    problem: LinearProblem
    arc_variables: Mapping[tuple[str, str, str], int]
    visit_variables: Mapping[tuple[str, str], int]
    order_variables: Mapping[tuple[str, str], int]
    time_variables: Mapping[tuple[str, str], int]
    big_m_time: float | None


@dataclass(frozen=True, slots=True)
class VRPMILPSolveResult:
    solution: VRPSolution | None
    validation: VRPValidationReport | None
    core_result: SolveResult


def _time_horizon(instance: VRPInstance) -> tuple[float, float]:
    if instance.travel_times is None:
        raise VRPCompileError("time-constrained MILP requires explicit travel times")
    max_edge = max(max(row) for row in instance.travel_times.values)
    max_service = max((c.service_duration for c in instance.customers), default=0.0)
    markers = [0.0]
    for c in instance.customers:
        if c.time_window is not None:
            markers.extend((c.time_window.start, c.time_window.end))
    for v in instance.vehicles:
        markers.append(v.start_time)
        if v.latest_return is not None:
            markers.append(v.latest_return)
        if v.max_route_duration is not None:
            markers.append(v.start_time + v.max_route_duration)
    route_bound = (len(instance.customers) + 1) * max_edge + sum(c.service_duration for c in instance.customers)
    horizon = max(markers) + route_bound + 1.0
    big_m = horizon + max_edge + max_service + 1.0
    if not isfinite(horizon) or not isfinite(big_m):
        raise VRPCompileError("time horizon is not finite")
    return horizon, big_m


def compile_vrp_milp(instance: VRPInstance) -> VRPMILPCompilation:
    customers = tuple(c.id for c in instance.customers)
    n = len(customers)
    arc_vars: dict[tuple[str, str, str], int] = {}
    visit_vars: dict[tuple[str, str], int] = {}
    order_vars: dict[tuple[str, str], int] = {}
    time_vars: dict[tuple[str, str], int] = {}
    domains: list[VariableDomain] = []
    lower: list[float] = []
    upper: list[float] = []
    objective: list[float] = []

    def add_var(domain: VariableDomain, lo: float, hi: float, cost: float = 0.0) -> int:
        idx = len(domains)
        domains.append(domain); lower.append(float(lo)); upper.append(float(hi)); objective.append(float(cost))
        return idx

    cmap = instance.customer_map
    # Per-vehicle path variables on a virtual source/sink graph.
    for vehicle in instance.vehicles:
        vid = vehicle.id
        # unused vehicle: source -> sink with zero cost
        arc_vars[(vid, SOURCE, SINK)] = add_var(VariableDomain.BINARY, 0.0, 1.0, 0.0)
        for cid in customers:
            c = cmap[cid]
            arc_vars[(vid, SOURCE, cid)] = add_var(
                VariableDomain.BINARY, 0.0, 1.0, instance.distance(vehicle.start_node_id, c.node_id)
            )
            arc_vars[(vid, cid, SINK)] = add_var(
                VariableDomain.BINARY, 0.0, 1.0, instance.distance(c.node_id, vehicle.end_node_id)
            )
        for i in customers:
            for j in customers:
                if i == j:
                    continue
                arc_vars[(vid, i, j)] = add_var(
                    VariableDomain.BINARY, 0.0, 1.0, instance.distance(cmap[i].node_id, cmap[j].node_id)
                )
        for cid in customers:
            visit_vars[(vid, cid)] = add_var(VariableDomain.BINARY, 0.0, 1.0)
            order_vars[(vid, cid)] = add_var(VariableDomain.CONTINUOUS, 1.0, float(max(1, n)))

    horizon: float | None = None
    big_m: float | None = None
    if instance.has_time_constraints:
        horizon, big_m = _time_horizon(instance)
        for vehicle in instance.vehicles:
            for cid in customers:
                time_vars[(vehicle.id, cid)] = add_var(VariableDomain.CONTINUOUS, 0.0, horizon)

    row_i: list[int] = []
    col_i: list[int] = []
    data: list[float] = []
    cl: list[float] = []
    cu: list[float] = []
    row = 0

    def add_row(terms: Mapping[int, float], lo: float = -np.inf, hi: float = np.inf) -> None:
        nonlocal row
        for idx, coef in terms.items():
            if coef:
                row_i.append(row); col_i.append(idx); data.append(float(coef))
        cl.append(float(lo)); cu.append(float(hi)); row += 1

    # Every customer is assigned to exactly one vehicle.
    for cid in customers:
        add_row({visit_vars[(v.id, cid)]: 1.0 for v in instance.vehicles}, 1.0, 1.0)

    for vehicle in instance.vehicles:
        vid = vehicle.id
        # Exactly one source exit and exactly one sink entry; source->sink means unused.
        source_terms = {arc_vars[(vid, SOURCE, SINK)]: 1.0}
        source_terms.update({arc_vars[(vid, SOURCE, cid)]: 1.0 for cid in customers})
        add_row(source_terms, 1.0, 1.0)
        sink_terms = {arc_vars[(vid, SOURCE, SINK)]: 1.0}
        sink_terms.update({arc_vars[(vid, cid, SINK)]: 1.0 for cid in customers})
        add_row(sink_terms, 1.0, 1.0)

        for cid in customers:
            incoming = {arc_vars[(vid, SOURCE, cid)]: 1.0, visit_vars[(vid, cid)]: -1.0}
            outgoing = {arc_vars[(vid, cid, SINK)]: 1.0, visit_vars[(vid, cid)]: -1.0}
            for other in customers:
                if other == cid:
                    continue
                incoming[arc_vars[(vid, other, cid)]] = 1.0
                outgoing[arc_vars[(vid, cid, other)]] = 1.0
            add_row(incoming, 0.0, 0.0)
            add_row(outgoing, 0.0, 0.0)

        # Capacity is a route-level resource in CVRP/VRPTW.
        add_row({visit_vars[(vid, c.id)]: c.demand for c in instance.customers}, -np.inf, vehicle.capacity)

        # MTZ eliminates customer-only subtours disconnected from the virtual source/sink path.
        if n > 1:
            for i in customers:
                for j in customers:
                    if i == j:
                        continue
                    add_row(
                        {
                            order_vars[(vid, i)]: 1.0,
                            order_vars[(vid, j)]: -1.0,
                            arc_vars[(vid, i, j)]: float(n),
                        },
                        -np.inf,
                        float(n - 1),
                    )

        if instance.has_time_constraints:
            if big_m is None:
                raise VRPCompileError("time-constrained routing model requires a computed finite big-M")
            start = vehicle.start_time
            for cid in customers:
                customer = cmap[cid]
                t = time_vars[(vid, cid)]
                y = visit_vars[(vid, cid)]
                if customer.time_window is not None:
                    # t >= open - M(1-y); t <= close + M(1-y)
                    add_row({t: 1.0, y: -big_m}, customer.time_window.start - big_m, np.inf)
                    add_row({t: 1.0, y: big_m}, -np.inf, customer.time_window.end + big_m)
                travel = instance.travel_time(vehicle.start_node_id, customer.node_id)
                x = arc_vars[(vid, SOURCE, cid)]
                # t >= start + travel - M(1-x)
                add_row({t: 1.0, x: -big_m}, start + travel - big_m, np.inf)
            for i in customers:
                ci = cmap[i]
                for j in customers:
                    if i == j:
                        continue
                    travel = instance.travel_time(ci.node_id, cmap[j].node_id)
                    # t_j >= t_i + service_i + travel - M(1-x_ij)
                    add_row(
                        {
                            time_vars[(vid, j)]: 1.0,
                            time_vars[(vid, i)]: -1.0,
                            arc_vars[(vid, i, j)]: -big_m,
                        },
                        ci.service_duration + travel - big_m,
                        np.inf,
                    )
            deadlines = []
            if vehicle.latest_return is not None:
                deadlines.append(vehicle.latest_return)
            if vehicle.max_route_duration is not None:
                deadlines.append(vehicle.start_time + vehicle.max_route_duration)
            if deadlines:
                deadline = min(deadlines)
                for cid in customers:
                    c = cmap[cid]
                    travel = instance.travel_time(c.node_id, vehicle.end_node_id)
                    # t_c + service + travel <= deadline + M(1-x_cT)
                    add_row(
                        {time_vars[(vid, cid)]: 1.0, arc_vars[(vid, cid, SINK)]: big_m},
                        -np.inf,
                        deadline + big_m - c.service_duration - travel,
                    )

    A = sparse.csr_matrix((data, (row_i, col_i)), shape=(row, len(domains)), dtype=np.float64)
    problem = LinearProblem.from_data(
        A=A,
        c=np.asarray(objective, dtype=np.float64),
        variable_lower=np.asarray(lower, dtype=np.float64),
        variable_upper=np.asarray(upper, dtype=np.float64),
        constraint_lower=np.asarray(cl, dtype=np.float64),
        constraint_upper=np.asarray(cu, dtype=np.float64),
        domains=domains,
        objective_sense=ObjectiveSense.MINIMIZE,
        name=f"vrp-arc-flow:{instance.name}",
        metadata={
            "application": "routing",
            "formulation": "multi-vehicle-arc-flow-mtz",
            "n_customers": n,
            "n_vehicles": len(instance.vehicles),
            "time_constraints": instance.has_time_constraints,
        },
    )
    return VRPMILPCompilation(
        problem,
        MappingProxyType(dict(arc_vars)),
        MappingProxyType(dict(visit_vars)),
        MappingProxyType(dict(order_vars)),
        MappingProxyType(dict(time_vars)),
        big_m,
    )


def decode_vrp_milp_solution(instance: VRPInstance, compilation: VRPMILPCompilation, x: np.ndarray) -> tuple[VRPRoute, ...]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != compilation.problem.n_variables or not np.isfinite(values).all():
        raise VRPCompileError("MILP candidate vector has invalid shape or non-finite values")
    canonical = validate_solution(compilation.problem, CandidateSolution(x=values))
    if not canonical.valid:
        raise VRPCompileError("MILP candidate failed canonical formulation validation: " + "; ".join(canonical.warnings))
    routes: list[VRPRoute] = []
    all_customers = set(c.id for c in instance.customers)
    decoded_customers: set[str] = set()
    for vehicle in instance.vehicles:
        vid = vehicle.id
        selected = {(tail, head) for (v, tail, head), idx in compilation.arc_variables.items() if v == vid and values[idx] > 0.5}
        outgoing: dict[str, str] = {}
        incoming: dict[str, str] = {}
        for tail, head in selected:
            if tail in outgoing or head in incoming:
                raise VRPCompileError(f"vehicle {vid!r} candidate has duplicate incoming/outgoing arcs")
            outgoing[tail] = head; incoming[head] = tail
        if SOURCE not in outgoing or SINK not in incoming:
            raise VRPCompileError(f"vehicle {vid!r} candidate lacks a source-to-sink path")
        nxt = outgoing[SOURCE]
        if nxt == SINK:
            if len(selected) != 1:
                raise VRPCompileError(f"unused vehicle {vid!r} contains extra selected arcs")
            routes.append(VRPRoute(vid, ()))
            continue
        customer_ids: list[str] = []
        seen: set[str] = set()
        while nxt != SINK:
            if nxt not in all_customers or nxt in seen:
                raise VRPCompileError(f"vehicle {vid!r} candidate contains an invalid customer path")
            seen.add(nxt); customer_ids.append(nxt); decoded_customers.add(nxt)
            if nxt not in outgoing:
                raise VRPCompileError(f"vehicle {vid!r} path terminates before the sink")
            nxt = outgoing[nxt]
        if len(selected) != len(customer_ids) + 1:
            raise VRPCompileError(f"vehicle {vid!r} contains a disconnected subtour")
        routes.append(VRPRoute(vid, customer_ids))
    if decoded_customers != all_customers:
        raise VRPCompileError("decoded MILP routes do not cover every customer exactly once")
    return tuple(routes)


def solve_vrp_milp(instance: VRPInstance, *, backend=None) -> VRPMILPSolveResult:
    compilation = compile_vrp_milp(instance)
    core = execute(compilation.problem, ScipyHighsBackend() if backend is None else backend)
    if core.x is None:
        return VRPMILPSolveResult(None, None, core)
    if core.validation is None or not core.validation.valid:
        raise VRPCompileError("VRP MILP backend returned a candidate that failed SolverPilot core validation")
    routes = decode_vrp_milp_solution(instance, compilation, core.x)
    objective = sum(route_distance(instance, route) for route in routes)
    solution = VRPSolution(
        routes,
        objective,
        method="milp-arc-flow-mtz",
        is_exact=True,
        optimality_proven=core.optimality_evidence.independently_verified_optimal,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        metadata={
            "backend": core.trace.backend,
            "backend_status": core.backend_status,
            "backend_reported_optimal": core.optimality_evidence.backend_reported_optimal,
            "independent_optimality_proof": core.optimality_evidence.independently_verified_optimal,
        },
    )
    report = validate_vrp_solution(instance, solution)
    if not report.valid:
        raise VRPCompileError("decoded VRP MILP solution failed independent routing validation: " + "; ".join(report.errors))
    return VRPMILPSolveResult(solution, report, core)
