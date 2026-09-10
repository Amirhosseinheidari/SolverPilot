# M29 — Value-of-Information / Feature Representation Audit

Status: protocol frozen before **new feature extraction and feature-family modelling** on 2026-09-06.

## Claim boundary

M29 is a post-outcome representation audit. The M25 and M28 solver-performance outcomes already exist and are known historically. Therefore M29 is **not** fresh held-out promotion evidence and can never enable production performance routing by itself.

Its only decision is whether there is enough stable, cost-effective information in richer feature representations to justify one final fresh-corpus selector experiment (M30). If the audit fails, the learned LP-routing branch closes for 1.0.

## Literature-derived design

The feature families are fixed from prior literature rather than chosen after feature/target correlations are observed:

1. **A — base structure**: size, density, aspect ratio.
2. **B — canonical static**: row/column sparsity distributions, objective/RHS/bound statistics, coefficient scaling/dynamism, sign structure and simple constraint/bound composition. This mirrors the families used in MIPLIB 2017's data-driven feature space, adapted to the actual continuous LP IR.
3. **C — topology/decomposition proxies**: bipartite connected-component statistics and concentration measures for row/column degrees. This is an inexpensive structural proxy for decomposition/blockiness.
4. **D — bounded solver-native probe**: HiGHS presolve only; no simplex/IPM solve, no iteration counts from either candidate method. Features are presolved row/column/nonzero reduction ratios and probe cost. This is a probing/landmarking family and is explicitly version-specific.

Feature cost is part of every cost-adjusted diagnostic. Parse/ingestion cost is excluded because the production planner receives an already constructed canonical IR; additional feature/probe work is included.

## Source cohorts

- M25: 48 public/OOD MIPLIB-derived LP relaxations, two repeated comparative SciPy/HiGHS DS/IPM campaigns.
- M28: 48 fresh public/OOD MIPLIB-derived LP relaxations with zero overlap with M25, three-round native HiGHS 1.10 simplex/IPM campaign.

No M26/M27 consumed held-out instance is reused as a promotion test. M25 and M28 are used only for representation discovery/audit.

## Targets

For each instance:

- `delta_s = ipm_cost_s - simplex_cost_s` (positive means simplex/DS is faster),
- `delta_relative = delta_s / ((ipm_cost_s + simplex_cost_s)/2)`,
- `absolute_opportunity_s = abs(delta_s)`.

M25 costs are made robust by taking the median of the two run-level median costs for each backend. M28 uses its three-round medians.

## Pre-registered analyses

### 1. Feature cost
For each cumulative family A, A+B, A+B+C, A+B+C+D report median/p95 cost and cost as a fraction of the target cohort's SBS and VBS opportunity.

### 2. Stable univariate information
For every feature, compute Spearman correlation with `delta_s` independently in M25 and M28. A feature is called *directionally stable* only when:

- correlation signs agree,
- `abs(rho) >= 0.25` in both cohorts,
- Benjamini-Hochberg FDR `q <= 0.10` in at least one cohort and raw permutation `p <= 0.10` in the other.

Permutation p-values use 2,000 fixed-seed permutations. This is discovery evidence, not a confirmatory p-value claim.

### 3. Cross-cohort low-capacity transfer probe
For each cumulative family, fit standardized ridge regression to `delta_relative` on one cohort and evaluate on the other, then reverse direction. Alpha is selected only on the source cohort from the fixed grid `{0.1, 1, 10, 100}` using 5-fold shuffled CV with a fixed seed. Report:

- target Spearman correlation,
- decisive winner balanced accuracy for instances with `abs(delta_relative) >= 0.05`,
- cost-adjusted policy/SBS if choosing simplex when predicted delta is positive and IPM otherwise,
- harmful-switch count and p90 absolute regret.

This probe is intentionally low-capacity. It does **not** authorize deployment.

## M30 authorization gate

M30 is authorized only if at least one cumulative feature family satisfies all of:

1. static/probe coverage >= 95% in both cohorts (family D may fail independently without invalidating A–C),
2. median cumulative feature cost <= 10% of median target SBS in both cohorts,
3. at least two directionally stable individual features in that family,
4. cross-cohort target Spearman >= 0.30 in both directions,
5. decisive balanced accuracy >= 0.60 in both directions,
6. cost-adjusted policy/SBS <= 0.99 in at least one direction and <= 1.01 in the reverse direction,
7. no use of simplex/IPM solve results or iteration trajectories in the feature values.

If no family passes, learned LP routing closes for OptiMind 1.0. If a family passes, M30 may use only the authorized feature family (or a strict subset) on a completely fresh public/OOD corpus with full provenance and one-time held-out evaluation.

## Non-goals

- no production selector training,
- no test-set promotion,
- no post-hoc threshold search on M27's consumed test,
- no claim that correlation implies causal algorithm suitability,
- no claim that HiGHS-presolve probe features transfer across solver versions.
