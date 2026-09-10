# OptiMind Track P — P2 Status

## Classification

**P2 PARAMETER COMPILER / CACHE / INCREMENTAL RECOMPILATION ENGINE — VERIFIED RELEASE**  
Package version: **0.0.33**

P2 is built on the verified P1 `0.0.32` semantic modeling kernel and the source-integrated P0 FROZEN 1.0.0 contract.

## Completed

- model-owned compile cache with semantic-template and execution-snapshot layers;
- value-only Parameter updates preserve SemanticHash and CompilationHash while changing DataHash;
- P0 RFC-004 compliant CompilationHash no longer embeds DataHash;
- Parameter dependency graph for constraints and objective;
- partial recompilation of only affected semantic constraint blocks;
- objective-only recompilation without re-scalarizing unrelated constraints;
- bounded LRU execution snapshot cache (`8` snapshots per semantic template);
- cached snapshot revisit path;
- execution mutation classification using existing Track-M mutation taxonomy;
- parameter map now exposes affected semantic/execution regions;
- explicit `cache_status`, changed parameters, row reuse/recompile counts and execution mutations in CompilationReport;
- `Model.clear_compile_cache()`, `Model.compiler_cache_info`, and `CompiledModel.refresh()`;
- structural-degree target selection: parameter values cannot silently switch LP/QP target;
- structurally quadratic constraints fail closed even when current coefficients evaluate to zero;
- QP parameter updates re-check convexity when the quadratic objective changes;
- legacy direct LP/QP APIs remain unchanged and bypass semantic compiler overhead;
- full P1/M26 regression, randomized cached-vs-full equivalence campaigns, numerical solve cross-checks, offline wheel and isolated smoke.

## Explicitly deferred

- backend-bound persistent session patching and solver state reuse;
- Bridge/Reformulation Engine and plan-sensitive certificates;
- backend capability protocol v2;
- conic/NLP/MINLP/CP/routing layers;
- production selector changes.

## Evidence summary

- full runtime suite: **291 passed / 3 skipped / 0 failed / 0 errors**;
- P2-specific tests: **17 deterministic/cache tests + 3 property campaigns**;
- cached-vs-full property work: **400 global LP updates + 300 localized dependency updates + 180 convex QP updates = 880 updates**;
- cached solve campaign: **100/100 LP cached-vs-full objectives**, **120/120 QP cached-vs-analytic optima**;
- maximum QP primal error: `6.661338147750939e-16`;
- maximum QP objective error: `8.881784197001252e-16`.

## Retained performance evidence

Smoke timing on this host, not a production claim:

- localized single-RHS dependency update: cached/full median ratio ≈ `0.0254`;
- objective-only update: cached/full median ratio ≈ `0.0888`;
- global dense A+b update: cached/full median ratio ≈ `0.9823`;
- execution snapshot hit: median ≈ `8 µs`.

The global-dense result is intentionally retained: P2 does **not** claim that caching accelerates updates that require almost all execution data to be regenerated.
