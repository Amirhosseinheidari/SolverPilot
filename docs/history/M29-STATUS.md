# M29 STATUS — Value-of-Information / Feature Representation Audit

Version: **0.0.34**

Classification: **VERIFIED NEGATIVE REPRESENTATION RESULT — LEARNED LP ROUTING CLOSED FOR 1.0**

## Question

Do richer, literature-grounded, cheaply-computable LP features contain stable and transferable information about the **absolute DS-vs-IPM regret** that is strong enough to justify one final fresh-corpus selector experiment?

## Claim boundary

M29 is intentionally **post-outcome discovery evidence** over the already observed M25 and M28 cohorts. It is not fresh held-out promotion evidence and can never enable production performance routing by itself.

The only allowed M29 decision is whether the pre-registered gate authorizes a future fresh-corpus M30 selector experiment.

## Evidence integrity

- M25 instances: **48**.
- M28 instances: **48**.
- M25/M28 overlap: **0**.
- Feature extraction: **48/48 + 48/48 complete**.
- HiGHS presolve probe: **48/48 + 48/48 complete**.
- Source/protocol integrity checks: **all pass**.
- Canonical scaling property campaign: **200/200 pass**, 0 failures.

## Feature families

- **A — base structure:** 5 features.
- **B — canonical static:** 143 additional features.
- **C — topology/decomposition proxies:** 7 additional features.
- **D — bounded HiGHS presolve probe:** 6 additional features.
- Total: **161** features.

Rows and objective coefficients are normalized by max-absolute coefficient before representation-sensitive statistics, matching the MIPLIB 2017 representation principle.

## Stable univariate information

Exactly **1 / 161** feature passed the pre-registered directional-stability criterion:

`C:row_degree_max_over_mean`

- M25 Spearman: **0.299**.
- M28 Spearman: **0.452**.

But after residualizing against the five base size/density features:

- M25 residual rho: **-0.011**.
- M28 residual rho: **0.197**.

Thus the only stable feature is largely a proxy for basic scale/sparsity rather than clearly independent transferable information.

## Feature cost

Median cumulative feature cost as a fraction of median SBS instance time:

| family | M25 | M28 |
|---|---:|---:|
| A | 0.10% | 0.07% |
| A+B | 13.07% | 9.79% |
| A+B+C | 16.77% | 17.39% |
| A+B+C+D | 41.97% | 65.82% |

Only family A passes the pre-registered <=10% cost gate in both cohorts. Rich static/topology/probing families are too expensive in the current Python implementation for short-to-medium LP solves.

## Cross-cohort transfer

| source→target | family | target rho | decisive BAcc | policy/SBS, zero feature cost | policy/SBS, real feature cost |
|---|---|---:|---:|---:|---:|
| m25→m28 | A | -0.066 | 0.772 | 1.216 | 1.217 |
| m28→m25 | A | -0.053 | 0.371 | 1.330 | 1.330 |
| m25→m28 | AB | 0.251 | 0.772 | 1.054 | 1.126 |
| m28→m25 | AB | 0.085 | 0.553 | 1.255 | 1.332 |
| m25→m28 | ABC | 0.275 | 0.772 | 1.054 | 1.161 |
| m28→m25 | ABC | 0.074 | 0.553 | 1.255 | 1.361 |
| m25→m28 | ABCD | 0.135 | 0.741 | 1.115 | 1.500 |
| m28→m25 | ABCD | 0.057 | 0.537 | 1.255 | 1.506 |

No cumulative family reaches Spearman >=0.30 in both directions. More importantly, **every transfer policy is worse than SBS even when feature cost is artificially set to zero**.

Therefore extractor overhead is not the sole cause of failure.

## Runtime-label stability audit

Timing noise is present but is not a sufficient explanation:

- M25 run1-vs-run2 delta Spearman: **0.936**.
- M25 winner-sign agreement: **91.67%**.
- M28 median pairwise round-delta Spearman: **0.922**.
- M28 unanimous three-round winner fraction: **81.25%**.

The VBS opportunity is also not a single-outlier artifact: the largest instance contributes 15.43% of M25 opportunity and 42.77% of M28 opportunity.

## Gate result

`m30_authorized = false`

No family passes all pre-registered M30-authorization gates.

## Production decision

- Learned LP performance routing remains **OFF**.
- No M30 selector experiment is authorized from M29.
- The learned LP-routing research branch is **closed for OptiMind 1.0**.
- Production keeps the conservative evidence-aware baseline.
- Future LP selector research requires a materially new representation/data regime, not post-hoc tuning on M25/M28/M27.

## Next milestone

Proceed to **Release Consolidation / API Freeze**, not another learned selector experiment.

## Release verification

- Full regression: **267 tests / 264 passed / 3 skipped / 0 failures / 0 errors**.
- M29-focused suite: **8/8 passed**.
- Wheel: `optimind_core_codename-0.0.34-py3-none-any.whl`.
- Wheel SHA-256: `0f2a6cc83a8d4e854f1ab171c38eeedc7e368638a01cda54b284de36cf1a0048`.
- Isolated wheel import outside source tree: **PASS**.
- Isolated production smoke: conservative `scipy-highs-ds`, `VALID_OPTIMAL`, independent validation **PASS**.
- M29 evidence remains `comparative_validation`, `heldout=false`, `performance_ranking_validated=false`, `supports_performance_ranking=false`.

