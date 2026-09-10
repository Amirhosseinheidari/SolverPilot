# Benchmark Report — M4 LP Structural Holdout

**Date:** 2026-08-31  
**Scope:** controlled synthetic continuous LP; HiGHS dual simplex vs HiGHS IPM through SciPy development bridges.  
**Not a public benchmark or market claim.**

## Question

Does the current two-algorithm LP portfolio contain enough complementary behavior for an instance-based selection policy to have measurable opportunity after inspection overhead?

## Protocol

- Families: `tall_moderate`, `tall_strong`, `wide_moderate`, `balanced_sparse`, `balanced_dense`, `equality_heavy`.
- 4 seeds per family, 24 total instances.
- 3 timing repetitions per solver/instance; median used as cost.
- DS/IPM execution order alternates by seed.
- Solver outputs must both be independently validated and objectives must agree.
- Selection-rule candidates are fixed in benchmark source.
- Leave-one-family-out: rule is selected on the other five families and applied to the unseen family.
- Policy cost includes median inspector wall time.
- The production planner is **not modified** by this benchmark.

## Run 1

| Metric | Value |
|---|---:|
| DS mean cost | 0.112833 s |
| IPM mean cost | 0.142238 s |
| SBS | scipy-highs-ds |
| SBS cost | 0.112833 s |
| VBS cost | 0.102789 s |
| Policy cost incl. inspection | 0.103789 s |
| SBS→VBS gap closure | 0.901 |
| Policy/SBS mean ratio | 0.920 |
| Bootstrap 95% CI of ratio | [0.872, 0.971] |
| DS wins | 13 |
| IPM wins | 11 |

## Run 2 — unchanged code/seeds

| Metric | Value |
|---|---:|
| DS mean cost | 0.112333 s |
| IPM mean cost | 0.141417 s |
| SBS | scipy-highs-ds |
| SBS cost | 0.112333 s |
| VBS cost | 0.102739 s |
| Policy cost incl. inspection | 0.103599 s |
| SBS→VBS gap closure | 0.910 |
| Policy/SBS mean ratio | 0.922 |
| Bootstrap 95% CI of ratio | [0.877, 0.971] |
| DS wins | 13 |
| IPM wins | 11 |

Both confidence intervals remain below 1.0. The per-family winner pattern is identical across the two full runs.

## Structural result

Run 2 median family costs and winner counts:

| Family | DS median | IPM median | DS wins | IPM wins | Policy IPM choices |
|---|---:|---:|---:|---:|---:|
| tall_moderate | 0.182435 | 0.141386 | 0 | 4 | 4 |
| tall_strong | 0.099588 | 0.084777 | 0 | 4 | 4 |
| wide_moderate | 0.092613 | 0.088298 | 1 | 3 | 0 |
| balanced_sparse | 0.116128 | 0.160140 | 4 | 0 | 0 |
| balanced_dense | 0.089133 | 0.175163 | 4 | 0 | 0 |
| equality_heavy | 0.094150 | 0.192044 | 4 | 0 | 0 |


The LOFO procedure selected the same research rule in every fold: `ipm_if_m_over_n_ge_1.50`. This rule is **not promoted to production** because the benchmark families are synthetic and generated from the same codebase.

The wide family illustrates why this policy is not an oracle: IPM won 3/4 wide instances in both runs, while the tall-only threshold selected DS for that family. The policy therefore leaves part of the VBS opportunity uncaptured. This is useful evidence against overfitting the rule further before public data is available.

## Correctness

Maximum DS/IPM objective absolute difference was `0.0` in run 2. Every accepted solve passed canonical validation.

## Interpretation

What the result supports:
- DS and IPM have reproducibly complementary performance on these structural families;
- a cheap structural selector can outperform the synthetic SBS after paying inspection overhead;
- algorithm selection deserves continued research.

What it does not support:
- cross-solver superiority;
- public benchmark superiority;
- a production `m/n >= 1.5` heuristic;
- an AI/ML claim;
- generalization to MIP/QP/customer problems.

Raw artifacts:
- `benchmarks/results/m4-lp-selector-holdout-run1.json`
- `benchmarks/results/m4-lp-selector-holdout-run2.json`
- `benchmarks/results/m4-lp-selector-holdout.json` (latest run)
