# M2 Status — Native Adapter Contracts, Reuse Plumbing, and Portfolio Evaluation

**Date:** 2026-08-31  
**Version:** 0.0.3  
**Status:** M2 research/engineering milestone completed within the current runtime constraints.

## What M2 actually completed

### Portfolio evaluation layer

Implemented reusable metrics for lower-is-better solver portfolios:

- Single Best Solver (SBS);
- Virtual Best Solver (VBS);
- policy cost with policy/inspection overhead included;
- SBS→VBS gap closure;
- policy wins/ties;
- Dolan–Moré style finite-cost performance profiles.

These metrics are unit tested and used by the M2 synthetic portfolio experiments.

### Native adapter code paths

Implemented optional, import-safe adapter code for:

- `highspy` (`HighspyNativeBackend`) for LP, convex QP, and MILP model construction/status normalization;
- `osqp` (`OSQPNativeBackend`) for convex QP, persistent same-sparsity updates, and OSQP automatic warm-start semantics;
- `pyscipopt` (`PySCIPOptNativeBackend`) for LP/MILP and IIS extraction via `Model.generateIIS`.

**Important:** the native Python packages are not installed in the current execution environment and attempts to install them failed because outbound package resolution/download is unavailable. Therefore these adapter paths are **implemented but not native-integration-verified here**. Their integration tests exist and are skipped when the dependency is absent.

The default registry registers an optional native adapter only when its real dependency is importable. It does not expose stubs as available solvers.

### Session/reuse plumbing

`Session` now holds a persistent backend registry across solves. This is necessary for a stateful native backend (notably OSQP) to preserve internal solver state between revisions.

Reuse is not inferred from a capability declaration alone. `Session.last_reuse_assessment.applied` is set true only when the backend execution result explicitly reports that reuse was applied. An end-to-end stateful fake-backend contract test verifies this behavior.

### OSQP translation invariants

The canonical QP → OSQP conversion is implemented as:

- upper triangular `P` in CSC;
- canonical row constraints plus identity rows for variable bounds;
- `q`, `l`, `u` vectors;
- on identical structural hash, numerical updates use the documented `update(q=..., l=..., u=..., Px=..., Ax=...)` path.

A randomized property program checked **1200/1200** same-structure translation invariants over 200 seeds, including CSC sparsity/order stability needed by value-only `Px/Ax` updates.

### Optional dependency/extras policy

`pyproject.toml` now defines optional extras:

- `highs` → `highspy>=1.15.1,<2`
- `osqp` → `osqp>=1.1.3,<2`
- `scip` → `pyscipopt>=6.2.1,<7`
- `benchmark` → `benchopt>=1.9.1,<2`, `scikit-learn>=1.5,<2`
- `open-source` → highspy + OSQP + PySCIPOpt

Base installation remains NumPy/SciPy-only.

## Verification performed

Current full suite:

```text
83 passed, 3 skipped
```

The three skips are deliberately dependency-gated native integration suites:

1. native highspy integration;
2. native OSQP integration/reuse;
3. native PySCIPOpt integration/IIS.

Additional executable checks:

```text
M1 mutation/hash properties:       1000 / 1000 passed
M2 OSQP translation properties:   1200 / 1200 passed
compileall:                         passed
editable offline install:           passed
public import/version smoke:         passed (0.0.3)
```

## Empirical portfolio findings

### Experiment A — small synthetic LP portfolio

48 synthetic LP instances, HiGHS dual simplex vs HiGHS IPM, repeated timing.

Latest recorded run:

- DS wins: 46
- IPM wins: 2
- DS mean solver cost: ~0.02431 s
- IPM mean solver cost: ~0.03642 s
- SBS: DS
- VBS: ~0.02427 s
- current automatic policy always selected DS
- policy cost including runtime path: ~0.02486 s

Because SBS and VBS are extremely close, the denominator of SBS→VBS opportunity is tiny; the resulting gap-closure number is unstable and strongly negative. This suite is therefore **not useful evidence for a selector-performance claim**. It is useful evidence that easy/small portfolios can have almost no selection opportunity and that orchestration overhead matters.

### Experiment B — family-separated selector research smoke

A second synthetic suite intentionally introduced solver complementarity and used disjoint generator families for train vs holdout test.

Train winners:

- DS: 4
- IPM: 4

Holdout winners:

- DS: 4
- IPM: 4

On the latest recorded 8-instance holdout:

- SBS cost: ~0.19208 s
- VBS cost: ~0.17554 s
- current capability-only auto policy cost: ~0.19578 s
- current auto gap closure: ~-22.39%
- benchmark-only depth-2 decision tree cost: ~0.18057 s
- research tree gap closure: ~69.60%

The fitted research rule in this tiny run was essentially a threshold on problem row count (`log10_m`). It is **not** shipped in the production planner and must not be interpreted as a general solver-selection rule.

This is only a synthetic smoke signal that instance-aware selection *can* exploit a real portfolio gap. It is not evidence of performance on market workloads, MIPLIB, or QPLIB.

## What remains incomplete

The following are **not completed** and must not be described as completed:

- native highspy integration execution in this environment;
- actual HiGHS basis reuse / MIP-start application;
- native OSQP setup/update/warm-start execution in this environment;
- native PySCIPOpt solve/IIS execution in this environment;
- SCIP native reoptimization;
- Benchopt installation/execution;
- actual MIPLIB 2017 benchmark instances;
- actual QPLIB convex-continuous instances;
- learned selector accepted into core;
- empirically validated production planner policy;
- public package name/legal/trademark clearance.

## M2 conclusion

M2 gives the project three things M1 did not have:

1. code-level native solver contracts that are dependency-gated rather than faked;
2. persistent state/reuse plumbing with explicit proof-of-application semantics;
3. an empirical SBS/VBS/gap-closure evaluation framework that can reject a bad planner rather than merely report solver times.

The next milestone is blocked primarily by **execution environment/data access**, not by lack of architecture: install the native solver extras, run their integration suites, then execute Benchopt-backed MIPLIB/QPLIB/repeated-QP experiments on a controlled machine.
