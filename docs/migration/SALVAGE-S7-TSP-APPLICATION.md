# SALVAGE-S7 — TSP Application Pack

Status: local migration candidate. This document is not a public release qualification.

## Scope

S7 adds `solverpilot.applications.tsp` as an application layer on top of the existing SolverPilot core. It does not add a second runtime, backend registry, benchmark model, or constraint-programming IR.

Included:

- immutable TSP node/instance/distance models;
- independent route and objective validation;
- exact tiny-instance brute-force reference solver;
- exact Held–Karp dynamic-programming reference solver;
- deterministic nearest-neighbor and multi-start nearest-neighbor heuristics;
- 2-opt local search with symmetric fast-delta and asymmetric full-cost recomputation;
- exact directed MTZ MILP compilation to the existing `LinearProblem` IR;
- exact integer-cost circuit compilation to the existing CP IR;
- safe decoding back through the canonical LP/MILP and CP validators.

Deliberately excluded:

- Christofides until metric preconditions and matching semantics are sufficient to support an approximation-ratio claim;
- legacy TSP registries, selector/workflow/report/CLI layers;
- metaheuristic sprawl;
- silent rounding of floating costs for CP-SAT.

## Trust boundaries

`is_exact=True` means the algorithm/formulation is exact for the represented TSP. It does not by itself mean that a particular backend result has an independently reconstructed optimality proof.

- brute-force and Held–Karp set `optimality_proven=True` because SolverPilot performs the exhaustive/reference computation itself;
- MTZ MILP preserves the core SolverPilot optimality-evidence semantics and does not upgrade a backend `optimal` status to independent proof;
- CP circuit results are independently marked proven only when solved by the built-in exhaustive `ReferenceCPBackend`;
- all decoded backend candidates must first pass canonical formulation validation and then independent TSP route/cost validation.

## Asymmetric 2-opt correction

The legacy implementation used the four-edge symmetric 2-opt delta for every matrix. That formula is not valid for asymmetric TSP because reversing a segment reverses all internal arc directions. S7 uses the fast four-edge delta only when the matrix is verified symmetric; otherwise each candidate route cost is recomputed from the directed matrix.

A fixed regression fixture demonstrates the legacy failure: the old implementation returned a route with reported cost 28 while independent recomputation was 110. S7 returns a route whose reported/recomputed objective agree.

## CP numeric policy

The CP IR has an integer linear objective. S7 therefore accepts CP compilation only when every distance is exactly integer-valued in binary64 and no selected-tour objective can exceed signed int64. Floating-cost TSP instances must use the MILP compiler; S7 never rounds them silently.

## Optional runtime qualification

OR-Tools/CP-SAT is optional. In the local S7 verification environment it is unavailable, so CP-SAT runtime execution is recorded as optional-runtime-unavailable, not as a pass. The circuit formulation is cross-checked against the exhaustive `ReferenceCPBackend`.
