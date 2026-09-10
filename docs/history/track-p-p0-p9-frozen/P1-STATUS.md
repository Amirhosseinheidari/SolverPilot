# OptiMind Track P — P1 Status

## Classification

**P1 SEMANTIC MODELING KERNEL — VERIFIED RELEASE**  
Package version: **0.0.32**

P1 is built on M26 `0.0.31` and the source-integrated P0 FROZEN 1.0.0 architecture contract.

## Completed

- semantic `Model`, `Variable`, `Parameter`, `Expression` DAG;
- stable model ownership and cross-model rejection;
- scalar/vector affine relation modeling;
- continuous/integer/binary variable domains;
- Parameter shape/sign validation and value-only update semantics;
- deterministic SemanticHash / DataHash split;
- expression shape, sign, polynomial degree and dependency metadata;
- LP/MILP compile to current `LinearProblem`;
- convex continuous QP compile to current `QuadraticProblem` using the existing half-`x^T P x` convention;
- `CompiledModel`, SourceMap, ParameterMap and CompilationReport P1 artifacts;
- legacy LP/QP import through `Model.from_problem()`;
- legacy matrix-first APIs remain first-class and bypass semantic compile;
- source regression, randomized property campaigns, exact MILP enumeration and analytic QP cross-checks;
- offline wheel, isolated install/smoke and package integrity.

## Explicitly deferred

- compile cache / incremental Parameter patching (P2);
- production Bridge Engine (later P milestone);
- indicator/SOS/PWL reformulation;
- conic/NLP/MINLP/CP/routing layers;
- performance selector changes.

## Evidence summary

- P1 focused tests: 31 passed.
- Randomized compile/property work embedded in tests: 300 affine models + 250 Parameter updates + 120 QP compilations.
- Independent numerical regression: 100/100 binary MILPs against complete enumeration; 120/120 diagonal convex QPs against analytic optima.
- Full M26 + P1 regression: 274 collected / 271 passed / 3 skipped / 0 failed / 0 errors.
- Isolated wheel smoke imports from installed target, not source tree.

## Retained negative evidence

Compiler smoke is not yet competitive for larger dense semantic models. The legacy direct path remains unchanged and has no P1 modeling overhead. No P1 compiler speedup claim is made.

## Wheel

`optimind_core_codename-0.0.32-py3-none-any.whl`  
SHA-256: `6c48bffdb373a0145ab526e2014d2a08a741dbf6f997c349f4310c274ef7399f`
