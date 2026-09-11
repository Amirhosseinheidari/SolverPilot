# SolverPilot Extended Modeling API — 0.2

Track P P0–P9 is merged into SolverPilot under explicit submodule namespaces. These APIs are available in version 0.2 but are **not** added to the M30-frozen 78-symbol `solverpilot.__all__` compatibility surface yet.

## `solverpilot.model`

Semantic `Model`, `Variable`, `Parameter`, symbolic expressions/atoms, constraint sets, compiler/cache artifacts, and `CompiledModel`.

Key guarantees:

- cross-model expressions fail closed;
- `SemanticHash` and `DataHash` remain separate;
- parameter-only changes can reuse semantic compilation state;
- LP/MILP and convex continuous QP lower into the canonical matrix-first IR;
- extended structures route to bridge/conic/NLP/MINLP lowering only when their structural preconditions hold.

## `solverpilot.capabilities`

Capability Protocol 2.0 adds granular capability keys, support/restriction modes, verification levels, version-bound evidence, runtime conformance, and conservative projection from the legacy `BackendManifest`.

Legacy declarations are never automatically upgraded to verified v2 evidence.

## `solverpilot.bridges`

Bridge registry, bridge policy, indicator lowering, precondition certificates, source mappings, and transformation tapes.

Current verified production bridge family is deliberately narrow: fixed-state simplification and finite-bound-certified affine Big-M indicator lowering.

## `solverpilot.session`

`PersistentSession` complements the stable lightweight `Session` API. It binds a semantic model to a backend lifecycle and chooses native patching only when the exact persistent/incremental capability evidence is verified.

## `solverpilot.conic`

`ConicProblem`, SOC/RSOC/PSD/exponential/power blocks, independent cone validation, and the direct Clarabel adapter. The legacy CasADi/SuperSCS verification path remains available.

Clarabel enables PSD and convex quadratic-conic solving after analytic runtime conformance checks. The legacy CasADi adapter keeps its fail-closed restrictions for those features.

## `solverpilot.nlp`

Smooth continuous NLP IR, automatic differentiation, NumPy-space validation, and the pinned CasADi 3.7.2/Ipopt verification path.

A successful generic NLP result is a local-optimal candidate after validation/KKT checks; it is not a global proof.

## `solverpilot.minlp`

Certified convex binary MINLP IR, validation, exact binary enumeration, and Outer Approximation.

Global proof is restricted to the supported convex-binary certificate scope and requires original-space validation plus bound closure. General integer/nonconvex MINLP remains unsupported.

## `solverpilot.cp`

Constraint-programming IR and model for integer/Boolean/interval variables, linear constraints, `AllDifferent`, `ExactlyOne`, allowed `Table`, `Element`, `Circuit`, `NoOverlap`, and `Cumulative`.

The dependency-free reference backend performs exhaustive enumeration up to a state budget. The optional CP-SAT adapter is version-locked to OR-Tools `9.15.6755` in this release candidate.
The CP-SAT adapter runs the native binding in an isolated worker process to avoid cross-library HiGHS ABI collisions; the parent process independently revalidates the returned assignment/objective.

## Stability

These extended submodule APIs are merged and regression-tested but remain explicit submodule surfaces. Their semantics are documented and fail-closed; promotion into the stable top-level `solverpilot.__all__` surface requires an explicit pre-1.0 API decision rather than happening implicitly during this merge.

## Version 0.2 additions

See [the 0.2 guide](../release/SOLVERPILOT-PUBLIC-DOCS-0.2.md) for direct Clarabel, exponential/power cones, owned reoptimization, named model helpers, lexicographic objectives, batch cancellation and numerical certificates. The legacy CasADi restrictions above remain specific to that adapter.
