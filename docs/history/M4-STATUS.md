# M4 Status — Structural Holdout Selection Evidence and Reproducibility Hardening

**Date:** 2026-08-31  
**Version:** 0.0.5  
**Status:** M4 engineering/research milestone completed within current network constraints.

## What M4 actually completed

### 1. Rebuilt from the immutable M3 ZIP and reverified baseline

The loose `/mnt/data/optimind_m3/work` directory available in the runtime contained only reports/results, so M4 was reconstructed from the original `optimind_m3.zip`, whose archive contains the full source/tests/benchmarks tree. The reconstructed project installed editable offline and the M3 baseline was re-run before M4 changes.

### 2. Native public package provenance lock

M4 records exact CPython 3.13 Linux x86-64 wheel filenames, official `files.pythonhosted.org` URLs, and SHA-256 digests for:

- `highspy 1.15.1`;
- `osqp 1.1.3`;
- `PySCIPOpt 6.2.1`.

`tools/verify_native_wheels.py` validates offline wheel files before installation. Exact byte size is enforced where an exact source value was available; otherwise SHA-256 is authoritative.

The current container still cannot retrieve those wheel binaries. A direct `pip` attempt failed at DNS resolution and a direct file-host download attempt also failed. Therefore the three public-native integration suites remain skipped, not passed.

### 3. Budget semantics improved without silent ignoring

`SolveBudget.threads` can now be applied to a dataclass backend that explicitly exposes a `threads` field (currently the public `HighspyNativeBackend` contract). Backends without an enforceable thread field still raise `NotImplementedError`. Memory budgets remain unimplemented rather than silently ignored.

### 4. Controlled structural LP holdout benchmark

M4 adds a new benchmark specifically designed to test whether the existing HiGHS dual-simplex vs IPM development portfolio has real complementarity.

Protocol:

- 6 structural LP families;
- 4 random seeds per family = 24 instances;
- 3 timing repeats per solver/instance;
- solver order alternated by seed;
- candidate selection rules fixed in source before timing outcomes;
- leave-one-family-out rule selection, so the test family is excluded from rule tuning;
- policy cost includes problem-inspection time;
- direct solver objectives must agree before an instance is accepted;
- SBS, VBS, gap closure, performance profile, and bootstrap policy/SBS ratio are reported.

Two complete benchmark executions were performed without changing code or seeds.

Run 1:
- SBS: `scipy-highs-ds`;
- DS wins / IPM wins: 13 / 11;
- gap closure: 0.901;
- policy/SBS mean-cost ratio: 0.920;
- bootstrap 95% CI: [0.872, 0.971].

Run 2:
- SBS: `scipy-highs-ds`;
- DS wins / IPM wins: 13 / 11;
- gap closure: 0.910;
- policy/SBS mean-cost ratio: 0.922;
- bootstrap 95% CI: [0.877, 0.971].

Winner pattern and policy choices were identical across both full reruns. This is the first positive selection evidence in the project, but it is deliberately classified as **synthetic within-HiGHS algorithm-selection evidence**, not a cross-solver or public-corpus claim.

### 5. M1/M2/M3 regression evidence re-run

After M4 changes:

- M1 mutation/hash properties: 1000 / 1000 passed;
- M2 OSQP translation properties: 1200 / 1200 passed;
- M3 native HiGHS repeated-LP benchmark re-ran successfully;
- every expected reuse step remained applied;
- every warm/cold objective still agreed.

Latest M3 median stateful/cold wall ratios after rerun:

- small: 0.631;
- medium: 0.418;
- large: 0.336.

These are timing results and therefore replace, rather than cherry-pick alongside, earlier measurements when describing the current runtime execution.

## Verification performed

```text
pytest:                         98 passed, 3 skipped
M1 mutation/hash checks:       1000 / 1000 passed
M2 OSQP translation checks:    1200 / 1200 passed
M4 selector benchmark:         24 instances × 2 solvers × 3 repeats, run twice
M4 objective disagreements:    0
M3 reuse regression:           passed
```

The three skips remain:
1. public `highspy` integration;
2. public `osqp` integration/reuse;
3. public `pyscipopt` integration/IIS.

## What M4 does NOT prove

M4 does not prove:

- that OptiMind beats a single best solver on MIPLIB, QPLIB, Netlib, or customer workloads;
- that the aspect-ratio rule should enter the production planner;
- that cross-solver selection is solved;
- that public `highspy 1.15.1`, OSQP 1.1.3, or PySCIPOpt 6.2.1 adapters execute correctly in the target package environment;
- that the observed ~8% synthetic policy improvement survives public/OOD corpora;
- that a learned selector is justified.

Production planner evidence level remains `capability_only`.

## Remaining high-priority gates

1. Execute exact public native packages from the locked wheels in a network-enabled/offline-wheel environment.
2. Run actual MIPLIB 2017 benchmark instances.
3. Run QPLIB continuous-convex subset.
4. Add a genuinely cross-solver portfolio (not only two algorithms from HiGHS).
5. Freeze train/validation/test manifests before any production performance policy is tuned.
6. Benchmark OSQP state reuse and SCIP IIS/reoptimization with their actual packages.

## M4 conclusion

M4 changes the selection story from “no observable opportunity” to “there is reproducible structural complementarity in the current DS/IPM portfolio.” The evidence is strong enough to justify continued selector research but not strong enough to enable an empirical production selector. Reoptimization remains the more mature product capability because it has native execution evidence and repeated-sequence speedups; selection now has a credible research signal that must survive public benchmark corpora.
