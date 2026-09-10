from __future__ import annotations

from time import perf_counter

from .model import VRPInstance
from .solution import VRPRoute, VRPSolution
from .validation import route_distance, validate_vrp_solution


def _partial_valid(instance: VRPInstance, routes: list[list[str]]) -> bool:
    # Check each route locally without requiring global coverage by constructing only capacity/timeline facts.
    customers = instance.customer_map
    for vehicle, ids in zip(instance.vehicles, routes):
        if sum(customers[c].demand for c in ids) > vehicle.capacity + 1e-9:
            return False
        if instance.has_time_constraints:
            from .timeline import build_route_timeline
            if not build_route_timeline(instance, VRPRoute(vehicle.id, ids)).valid:
                return False
    return True


def nearest_feasible_insertion(instance: VRPInstance, *, improve: bool = True) -> VRPSolution:
    started = perf_counter()
    routes: list[list[str]] = [[] for _ in instance.vehicles]
    unassigned: list[str] = []
    for customer in instance.customers:
        options: list[tuple[float, int, int, list[list[str]]]] = []
        for vi, vehicle in enumerate(instance.vehicles):
            for pos in range(len(routes[vi]) + 1):
                candidate_routes = [list(r) for r in routes]
                candidate_routes[vi].insert(pos, customer.id)
                if not _partial_valid(instance, candidate_routes):
                    continue
                before = route_distance(instance, VRPRoute(vehicle.id, routes[vi]))
                after = route_distance(instance, VRPRoute(vehicle.id, candidate_routes[vi]))
                options.append((after - before, vi, pos, candidate_routes))
        if not options:
            unassigned.append(customer.id)
            continue
        options.sort(key=lambda x: (x[0], x[1], x[2]))
        routes = options[0][3]
    if improve and not unassigned:
        routes = _local_search(instance, routes)
    route_objects = tuple(VRPRoute(v.id, ids) for v, ids in zip(instance.vehicles, routes))
    objective = sum(route_distance(instance, r) for r in route_objects)
    return VRPSolution(
        route_objects,
        objective,
        method="nearest-feasible-insertion+local-search" if improve else "nearest-feasible-insertion",
        is_exact=False,
        optimality_proven=False,
        unassigned_customer_ids=tuple(unassigned),
        runtime_seconds=perf_counter() - started,
        metadata={"heuristic": True},
    )


def _local_search(instance: VRPInstance, routes: list[list[str]], *, max_passes: int = 20) -> list[list[str]]:
    current = [list(r) for r in routes]
    for _ in range(max_passes):
        best = [list(r) for r in current]
        best_obj = sum(route_distance(instance, VRPRoute(v.id, ids)) for v, ids in zip(instance.vehicles, current))
        improved = False
        # Intra-route reversal; full recomputation is used so asymmetric costs are safe.
        for vi, ids in enumerate(current):
            for i in range(len(ids)):
                for j in range(i + 1, len(ids)):
                    cand = [list(r) for r in current]
                    cand[vi] = ids[:i] + list(reversed(ids[i : j + 1])) + ids[j + 1 :]
                    if not _partial_valid(instance, cand):
                        continue
                    obj = sum(route_distance(instance, VRPRoute(v.id, x)) for v, x in zip(instance.vehicles, cand))
                    if obj < best_obj - 1e-12:
                        best_obj, best, improved = obj, cand, True
        # Relocate one customer across any two routes (or within a route).
        for src in range(len(current)):
            for idx in range(len(current[src])):
                cid = current[src][idx]
                for dst in range(len(current)):
                    base_without = [list(r) for r in current]
                    base_without[src].pop(idx)
                    for pos in range(len(base_without[dst]) + 1):
                        cand = [list(r) for r in base_without]
                        cand[dst].insert(pos, cid)
                        if not _partial_valid(instance, cand):
                            continue
                        obj = sum(route_distance(instance, VRPRoute(v.id, x)) for v, x in zip(instance.vehicles, cand))
                        if obj < best_obj - 1e-12:
                            best_obj, best, improved = obj, cand, True
        current = best
        if not improved:
            break
    return current


def clarke_wright_savings(instance: VRPInstance) -> VRPSolution:
    """Classic parallel Clarke-Wright savings heuristic under explicit preconditions.

    S8 intentionally limits this implementation to the classical symmetric,
    single-depot, homogeneous-capacity CVRP.  It fails closed outside those
    assumptions rather than implying a guarantee for heterogeneous/asymmetric or
    time-window routing.
    """
    from .errors import VRPValidationError
    if instance.has_time_constraints:
        raise VRPValidationError("Clarke-Wright S8 heuristic is qualified only for CVRP without time constraints")
    starts={v.start_node_id for v in instance.vehicles}; ends={v.end_node_id for v in instance.vehicles}
    if len(starts)!=1 or starts!=ends:
        raise VRPValidationError("Clarke-Wright S8 heuristic requires one common start/end depot")
    capacities={v.capacity for v in instance.vehicles}
    if len(capacities)!=1:
        raise VRPValidationError("Clarke-Wright S8 heuristic requires homogeneous vehicle capacities")
    vals=instance.distances.values
    n=len(vals)
    for i in range(n):
        for j in range(i+1,n):
            if abs(vals[i][j]-vals[j][i])>1e-9:
                raise VRPValidationError("Clarke-Wright S8 heuristic requires a symmetric distance matrix")
    depot=next(iter(starts)); cap=next(iter(capacities)); cmap=instance.customer_map
    routes: dict[int,list[str]]={i:[c.id] for i,c in enumerate(instance.customers)}
    loads: dict[int,float]={i:c.demand for i,c in enumerate(instance.customers)}
    owner={c.id:i for i,c in enumerate(instance.customers)}
    savings=[]
    for a in instance.customers:
        for b in instance.customers:
            if a.id>=b.id: continue
            s=instance.distance(depot,a.node_id)+instance.distance(depot,b.node_id)-instance.distance(a.node_id,b.node_id)
            savings.append((-s,a.id,b.id))
    savings.sort()
    for neg_s,a,b in savings:
        # Once the fleet limit is met, do not accept a non-positive merge.
        # Before that point a non-positive merge may still be necessary to
        # reduce the number of routes to the available vehicle count.
        if -neg_s <= 1e-12 and len(routes) <= len(instance.vehicles):
            break
        ra=owner[a]; rb=owner[b]
        if ra==rb or ra not in routes or rb not in routes: continue
        A=routes[ra]; B=routes[rb]
        if loads[ra]+loads[rb] > cap+1e-9: continue
        candidates=[]
        # Classical endpoint merges, including route reversal because costs are symmetric.
        if A[-1]==a and B[0]==b: candidates.append(A+B)
        if A[0]==a and B[-1]==b: candidates.append(B+A)
        if A[0]==a and B[0]==b: candidates.append(list(reversed(A))+B)
        if A[-1]==a and B[-1]==b: candidates.append(A+list(reversed(B)))
        if not candidates: continue
        merged=min(candidates, key=lambda ids: (route_distance(instance, VRPRoute(instance.vehicles[0].id,ids)), tuple(ids)))
        routes[ra]=merged; loads[ra]+=loads[rb]
        del routes[rb]; del loads[rb]
        for cid in merged: owner[cid]=ra
    route_lists=sorted(routes.values(), key=lambda ids: (tuple(ids),len(ids)))
    if len(route_lists)>len(instance.vehicles):
        fallback = nearest_feasible_insertion(instance, improve=True)
        return VRPSolution(
            fallback.routes, fallback.objective, method="clarke-wright-savings:fallback-nearest",
            is_exact=False, optimality_proven=False,
            unassigned_customer_ids=fallback.unassigned_customer_ids,
            runtime_seconds=fallback.runtime_seconds,
            metadata={"qualified_scope":"symmetric-single-depot-homogeneous-cvrp","fallback":"nearest-feasible-insertion"},
        )
    assigned=route_lists; unassigned=()
    route_objects=[]
    for i,v in enumerate(instance.vehicles):
        route_objects.append(VRPRoute(v.id, assigned[i] if i<len(assigned) else ()))
    objective=sum(route_distance(instance,r) for r in route_objects)
    return VRPSolution(route_objects,objective,method="clarke-wright-savings",is_exact=False,optimality_proven=False,unassigned_customer_ids=unassigned,metadata={"qualified_scope":"symmetric-single-depot-homogeneous-cvrp"})
