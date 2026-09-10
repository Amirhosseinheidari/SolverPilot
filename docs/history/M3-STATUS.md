# M3 Status — Native HiGHS Reuse Verification and Trace Hardening

**Date:** 2026-08-31  
**Version:** 0.0.4  
**Status:** M3 engineering/research milestone completed within the current runtime constraints.

## What M3 actually completed

### 1. Real native HiGHS state/reuse execution via SciPy's vendored binary

The runtime cannot install public `highspy`, `osqp`, or `pyscipopt` wheels because outbound package traffic is blocked. M3 found a legitimate local verification path: SciPy 1.17.0 contains a private native HiGHS binding at `scipy.optimize._highspy._core` backed by HiGHS **1.8.0**.

A new **development-only** backend, `ScipyVendoredHighsDevBackend`, uses that private binary to test stateful semantics. It is deliberately **not registered by default** and must never become a production dependency because SciPy private APIs are unstable.

The backend has been executed end-to-end for:

- LP solve;
- convex QP solve;
- MILP solve;
- same-structure LP data mutation;
- explicit LP basis hot-start (`getBasis` → updates → `setBasis` → `run`);
- MILP previous-incumbent injection through `setSolution`;
- structural-change invalidation/cold rebuild;
- LP IIS extraction through native HiGHS `getIis`.

This converts several M2 concepts from "contract only" into **runtime-verified behavior on a real HiGHS native binary**.

It does **not** convert the optional public `highspy 1.15.1` integration test from skipped to passed. The locally vendored HiGHS is older (1.8.0), so public-package ABI/API compatibility remains a separate gate.

### 2. Stateful repeated-LP benchmark with basis reuse

A controlled benchmark now compares:

1. fresh HiGHS solver/model construction for every LP in a 10-step sequence;
2. one persistent HiGHS model with same-structure updates and explicit prior basis reuse.

Protocol:

- solver: HiGHS simplex;
- presolve: disabled to isolate basis-reuse mechanics;
- threads: 1;
- sequence: 10 related LPs;
- 3 problem sizes;
- 3 random seeds per size;
- 3 timing repetitions per seed;
- execution order alternated to reduce order/cache bias;
- one-time solver warm-up outside measurement;
- every stateful/cold objective compared for agreement;
- 5,000-resample bootstrap CI on median ratios.

Results (stateful / cold wall-time ratio; lower is better):

| Size | Median wall ratio | Bootstrap 95% CI | Approx. sequence speedup | Median simplex-iteration ratio |
|---|---:|---:|---:|---:|
| small (80×200) | 0.630 | [0.601, 0.688] | 1.59× | 0.145 |
| medium (150×400) | 0.421 | [0.414, 0.437] | 2.37× | 0.172 |
| large (250×700) | 0.346 | [0.330, 0.365] | 2.89× | 0.165 |

Across every measured sequence:

- objectives agreed within the benchmark tolerance;
- reuse was applied on exactly 9/9 expected post-initial steps;
- simplex iteration count fell to roughly 14.5–17.2% of cold solve totals.

This is **synthetic repeated-LP evidence**, not a market/SOTA benchmark and not a universal speedup claim.

### 3. Benchmark bug found and fixed

During M3, a real measurement bug was detected:

`Highs.getRunTime()` on a persistent solver is cumulative. Treating it as a per-solve runtime incorrectly made the warm path look slower.

The backend now records:

- `highs_run_delta_s` — per-run delta;
- `highs_runtime_cumulative_s` — cumulative native timer;
- outer wall time remains measured separately to include Python/model-update overhead.

A second performance issue was found in the update bridge: matrix-value comparison originally performed Python-level sparse scalar lookup per nonzero. It was replaced by vectorized CSR data comparison and only changed coefficients invoke `changeCoeff`.

Both corrections materially changed the end-to-end result and are retained in tests/benchmark code.

### 4. Reuse evidence promoted into the stable trace schema

Trace schema version is now **0.2** and contains first-class fields:

- `reuse_applied`;
- `reuse_mode`.

`Session` consumes this execution evidence rather than interpreting capability metadata as proof that reuse occurred.

Current native development modes observed:

- `cold_rebuild`;
- `explicit_basis_hot_start`;
- `mip_solution_start`.

### 5. HiGHS adapter translation compatibility strengthened

Even though public `highspy` is unavailable, the existing `HighspyNativeBackend` LP and QP model builders are passed into the real SciPy-vendored HiGHS binary during tests. The binary accepts both translations and solves them correctly.

This is stronger than a pure mock/unit test, but it is still **not** a substitute for installing and exercising public `highspy 1.15.1`.

### 6. IIS result typing

The development HiGHS IIS verifier now returns a typed `HighsDevIISResult` rather than leaking arbitrary private pybind objects. The type deliberately keeps bound-status codes solver-native because this development backend is not our stable diagnostic schema.

## Verification performed

Full suite after M3 changes:

```text
94 passed, 3 skipped
```

The three skips are still real missing-package integration tests:

1. public `highspy` integration;
2. public `osqp` integration/reuse;
3. public `pyscipopt` integration/IIS.

Additional checks:

```text
M1 mutation/hash properties:      1000 / 1000 passed
M2 OSQP translation properties:  1200 / 1200 passed
M3 native HiGHS reuse benchmark: 27 measured sequence runs
M3 expected reuse steps:          all matched
M3 objective cross-checks:        all matched
compileall:                       passed
```

## Network/package-install finding

M3 verified that the problem is not package availability:

- PyPI publishes a CPython 3.13 Linux x86-64 wheel for `highspy 1.15.1`;
- PyPI publishes a matching wheel for `osqp 1.1.3`;
- PyPI publishes a matching wheel for `PySCIPOpt 6.2.1`.

Package installation/download fails because the runtime cannot establish outbound connectivity to the file host, even when an IP is supplied directly. Therefore skipped public-native tests remain a genuine environment limitation.

## What remains incomplete

The following are **not completed**:

- public `highspy 1.15.1` end-to-end integration execution;
- public highspy basis/MIP-start reuse integration on the target package;
- native OSQP setup/update/automatic-warm-start execution;
- native PySCIPOpt solve/IIS execution;
- SCIP native reoptimization;
- Benchopt installation/execution;
- actual MIPLIB 2017 instance runs;
- actual QPLIB convex-continuous instance runs;
- market/workload benchmark suite;
- production solver-selection policy with held-out public evidence;
- learned selector accepted into core;
- public package-name/legal/trademark clearance.

## M3 conclusion

M3 gives the project the first **runtime proof that state reuse is not merely an architectural idea**. On a real HiGHS binary, the `Session`/persistent-backend design can safely carry a basis or incumbent across related solves, and on the controlled repeated-LP workload this produces a large reduction in simplex work and a positive end-to-end sequence improvement.

The key negative finding remains equally important: DS-vs-IPM selection evidence is still synthetic and public benchmark corpora have not been run. Reoptimization has stronger empirical support than the current solver-selector story. Until MIPLIB/QPLIB and native OSQP/SCIP runs exist, the product should emphasize **safe adaptive execution/reuse/diagnostics infrastructure**, not claim a generally superior automatic solver selector.
