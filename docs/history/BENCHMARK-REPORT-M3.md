# Benchmark Report — M3

**Date:** 2026-08-31  
**Scope:** controlled native reoptimization verification plus prior synthetic portfolio findings. No market/SOTA claim.

## Experiment 1 — repeated LP basis reuse

Command:

```bash
PYTHONPATH=src python benchmarks/m3_highs_reoptimization.py
```

The benchmark uses the native HiGHS 1.8.0 binary vendored inside SciPy 1.17.0 through a private development adapter. This is specifically for architecture verification; production code must use public solver APIs.

### Method

Each sequence contains 10 LPs with unchanged matrix sparsity and small changes to objective/RHS values. The first stateful solve is cold. The following nine solves update the existing model and explicitly restore the previous valid basis. The baseline creates a fresh solver/model for every step.

All runs use simplex, one thread, and presolve disabled to isolate basis hot-start behavior. There are three seeds and three timing repetitions at each size (9 measured sequence pairs per size, 27 total). Order alternates between cold-first/stateful-first. A separate warm-up occurs before measurement.

### Results

| Problem shape | Median stateful/cold wall ratio | 95% bootstrap CI | Equivalent median speedup | Median stateful/cold simplex-iteration ratio |
|---|---:|---:|---:|---:|
| 80 constraints × 200 variables | 0.630 | [0.601, 0.688] | 1.59× | 0.145 |
| 150 × 400 | 0.421 | [0.414, 0.437] | 2.37× | 0.172 |
| 250 × 700 | 0.346 | [0.330, 0.365] | 2.89× | 0.165 |

Native solver-run deltas improve more than wall time because Python/model-update overhead remains visible. That distinction is exactly why both timers are reported.

### Correctness checks

Every warm/cold solution objective agrees within benchmark tolerance. Every stateful run reports reuse on exactly the nine expected post-initial solves.

### Interpretation

The result supports a limited claim:

> For these related synthetic LP sequences, a persistent HiGHS simplex model with valid basis reuse materially reduces simplex work and end-to-end sequence time compared with rebuilding each solve.

It does **not** establish:

- the same speedup on arbitrary LPs;
- a benefit when modifications are large;
- a benefit when presolve would dramatically reduce a cold model;
- behavior of public highspy 1.15.1;
- a benefit for QP/MILP in general.

HiGHS documentation itself notes that basis hot starts are most effective for sequences of related LPs and that sufficiently large changes or strong presolve reductions can make a scratch/IPM solve preferable. That is precisely why OptiMind needs mutation-aware policy rather than an unconditional warm-start flag.

## Experiment 2 — solver-selection status

M2 already demonstrated two distinct facts:

1. on the 48-instance small synthetic LP suite, dual simplex almost dominated IPM and SBS≈VBS, giving essentially no useful algorithm-selection opportunity;
2. on an intentionally complementary family-separated synthetic smoke, a tiny held-out research tree captured part of the SBS→VBS gap, while the capability-only production policy did not.

M3 attempted a substantially larger generated DS/IPM stress suite, but that experiment exceeded the runtime execution window and produced no accepted result artifact. Therefore it is **not counted** as evidence.

Current selector conclusion remains:

- the evaluation machinery is valid;
- portfolio complementarity is necessary;
- current production planner has no validated public performance model;
- public MIPLIB/QPLIB data remains the next required gate.

## Raw artifacts

- `benchmarks/results/m3-highs-reoptimization.json`
- `benchmarks/results/m2-lp-portfolio.json`
- `benchmarks/results/m2-selector-research.json`
