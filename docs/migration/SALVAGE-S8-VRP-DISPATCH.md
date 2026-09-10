# SALVAGE-S8 — VRP / Dispatch Application Pack

Status: migration-stage implementation; public/PyPI release qualification remains external.

## Goal

S8 salvages the useful routing semantics from the legacy OptiMind VRP/dispatch tree without importing its milestone/governance framework or duplicate plugin/runtime layers. The canonical SolverPilot optimization core remains authoritative.

## Implemented scope

- CVRP and VRPTW application model.
- Heterogeneous vehicles with explicit capacity, start node, end node, optional shift window, and optional maximum route duration.
- Mandatory customer coverage with non-negative demand, service duration, and at most one continuous time window per customer.
- Separate distance and travel-time matrices. Distance is never silently treated as travel time.
- Independent route/solution validation for coverage, duplicate assignments, vehicle references, capacity, recomputed distance/objective, customer time windows, vehicle shift return, and maximum route duration.
- Deterministic route timeline with arrival, waiting, service start, departure, return time, and route duration.
- Operator-facing diagnostics derived from validation/timeline evidence. Suggestions are not proofs or automatic operational decisions.
- Exhaustive reference solver for small instances with explicit search-space guards.
- Nearest-feasible-insertion heuristic with full-cost local search; it never claims exactness or optimality.
- Exact multi-vehicle arc-flow MILP with virtual source/sink, route-level capacity, MTZ subtour elimination, and conditional time constraints for VRPTW.
- MILP candidates must pass SolverPilot's canonical `LinearProblem` validation before decoding and then pass an independent routing validation after decoding.

## Important semantics

- A vehicle with no assigned customers is *unused*: it incurs zero route distance and zero route duration. The MILP represents this with a virtual zero-cost source-to-sink arc.
- When a vehicle has a shift window, S8 uses the shift start as the fixed dispatch start and the shift end as the latest return. S8 does not currently optimize departure time as a separate decision.
- Customer time-window feasibility is defined on service start. Waiting before a window opens is allowed.
- Service durations affect time propagation whenever a time-constrained instance is present.
- `VRPSolution.optimality_proven=True` is reserved for the exhaustive reference solver. A solver-reported optimal MILP result remains solver-certified unless the SolverPilot core has independent optimality evidence.

## Explicitly out of scope for S8

- Pickup-and-delivery precedence/pairing.
- Split deliveries.
- Multiple disjoint time windows per customer.
- Break scheduling, driver regulations, skills, compatibility classes, stochastic travel time, real-time re-dispatch, or live map-provider calls.
- A CP-SAT/OR-Tools routing formulation. The MILP and exhaustive reference are the exact formulations qualified in S8.
- Automatic operational approval. Diagnostics are suggestions for human review.
- Any legacy release-gate, approval-dossier, evidence-pack, or milestone framework.

## Legacy semantics retained

The rewrite preserves the useful ideas demonstrated by legacy tests for instance modeling, solution structure, independent feasibility validation, route timeline/waiting, operator resolution, and stronger baseline routing. The legacy data models and plugin/runtime coupling are intentionally not ported.
