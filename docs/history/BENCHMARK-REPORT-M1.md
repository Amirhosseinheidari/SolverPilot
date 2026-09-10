# M1 Synthetic Benchmark Report

**Date:** 2026-08-31  
**Purpose:** smoke verification of runtime semantics and timing instrumentation.  
**This is not a solver leaderboard or market-performance claim.**

## Dataset

Synthetic, deterministic generators only:

| Class | Instances | Purpose |
|---|---:|---|
| LP | 9 | Compare two real HiGHS LP algorithms and planner/inspector overhead |
| MILP | 10 | Verify automatic routing to an integer-capable backend |
| Convex QP | 20 | Verify QP end-to-end path against analytic optima |

## LP result consistency

Each LP was solved independently with:

- `scipy.optimize.linprog(method="highs-ds")`;
- `scipy.optimize.linprog(method="highs-ipm")`.

Every returned candidate passed canonical validation.

Maximum absolute objective difference between DS and IPM over the nine instances:

```text
0.0
```

This is a correctness signal for the smoke corpus, not evidence of universal numerical equivalence.

## Timing observations

Median backend solve time on the nine synthetic LPs:

```text
HiGHS dual simplex: ~0.00787 s
HiGHS IPM:          ~0.01054 s
```

Planner timing over LP+MILP auto-solves:

```text
median: ~0.0000904 s
max:    ~0.0001289 s
```

Inspector timing:

```text
median: ~0.0004607 s
max:    ~0.0008840 s
```

These values are machine-, version-, and dataset-specific.

Critically, M1 does **not** turn the observed DS/IPM timing difference into a general planner rule. The dataset is much too small and synthetic for that.

## MILP routing

All 10 binary knapsack instances:

- were automatically routed to `scipy-highs-bridge` because the LP-only DS/IPM adapters are capability-incompatible with MILP;
- returned `VALID_OPTIMAL`;
- passed independent integrality and constraint validation.

## QP analytic cross-check

Twenty diagonal positive-definite QPs were generated such that the unconstrained optimum was analytically available and inside the variable bounds.

Maximum coordinate error of the SLSQP development bridge relative to the analytic optimum:

```text
4.4627441841971205e-07
```

The runtime still reports these as `VALID_FEASIBLE`, not `VALID_OPTIMAL`, because M1 does not yet provide a QP optimality-certificate path for this bridge.

## Raw results

`benchmarks/results/m1-smoke.json`

## What this benchmark does not answer

It does not establish:

- the best LP algorithm across real workloads;
- the best MILP solver;
- whether automatic selection beats SBS;
- whether a useful SBS→VBS gap exists;
- reoptimization speedup;
- OSQP vs other QP solver performance;
- performance on MIPLIB/QPLIB;
- production overhead at large scale.

Those remain future experiments.
