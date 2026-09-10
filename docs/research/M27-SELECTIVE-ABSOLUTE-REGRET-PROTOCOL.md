# M27 Selective Absolute-Regret LP Selector Protocol

**Status:** FROZEN BEFORE HELD-OUT TEST OUTCOMES ARE READ OR EXECUTED

## Scientific role
M27 is the second learned DS-vs-IPM selector milestone. M26 showed that a relative log-cost-ratio ridge model can incur catastrophic absolute regret even when many individual labels are correct. M27 therefore targets downstream wall-clock cost directly and adds an explicit reject/fallback mechanism.

The 12-row M26 validation split is no longer treated as unbiased validation because M27 design was informed by M26's validation failure. M27 combines the original train+validation 32 rows as **development data only**. The original 16-row test split remains the sole held-out promotion set and must be evaluated exactly once after this protocol and the model artifact are frozen.

## Literature-derived design constraints
- Algorithm selection is evaluated against the single-best solver (SBS), not classification accuracy.
- Feature/inspection cost is charged to every selector decision.
- A selector may abstain and fall back to the default route; coverage is traded for lower decision risk.
- Tail failures matter: the model targets **absolute saving**, so large-cost mistakes receive proportionally large squared error.
- Low capacity is required because only 32 development instances are available and portfolio-selection theory warns that model/portfolio complexity can overfit small instance sets.
- M27 does not claim formal conformal coverage. Its leaf bootstrap bound is an empirical development-time safety statistic only.

## Frozen data
- Cohort: exactly the M25 48-instance public/OOD MIPLIB-derived LP-relaxation cohort.
- Group-safe split SHA-256: `6e94069451a07a8ec7ef786bff70ec844afeedd5282c93258fe47615ab0415e6`.
- Development: original train + validation = 32 rows.
- Held-out test: original test = 16 rows. Test costs/outcomes are forbidden to the M27 training/freezing process.

## Model family
A single low-capacity CART regression tree predicts **absolute saving from switching away from the development SBS**.

Features (same cheap structural features as M26):
1. `log1p(n)`
2. `log1p(m)`
3. `log1p(nnz)`
4. `log(density)`
5. `log1p(m/n)`

Frozen hyperparameters:
- `max_depth = 2`
- `min_samples_leaf = 8`
- criterion = squared error
- random_state = 270027

The development SBS is computed from the 32 development rows only. For each development row the target is:

`default_solver_cost - alternate_solver_cost`

so positive values mean the alternate route is faster.

## Selective switching rule
Default to the development SBS. Switch to the alternate route only when **all** conditions hold:
1. predicted leaf mean absolute saving > 0.005 s;
2. deterministic one-sided 90% bootstrap lower bound of leaf mean saving > 0.005 s;
3. leaf contains at least 8 development samples;
4. for every feature actually used by the fitted tree, the query feature lies inside the observed min/max range of the selected training leaf (no switch under feature extrapolation).

All selector decisions pay measured inspection cost. The no-selector SBS baseline pays no inspection cost.

## Development audit
Development performance is computed with leave-one-group-out predictions. This is model-development evidence only, not a promotion claim.
Required before the test may be opened:
- 32/32 OOF rows accounted;
- group leakage = 0;
- objective mismatches in source data = 0;
- OOF policy / development SBS <= 0.99;
- at least 3 but no more than 12 alternate-route switches;
- harmful alternate switches (alternate solver slower than default before inspection) = 0;
- p90 OOF policy / default-route cost <= 1.10;
- model and protocol SHA-256 frozen before test execution.

## One-time held-out test gate
Only after the development gate and model freeze pass:
- execute exactly the frozen 16 test instances in a fresh process-isolated campaign;
- two interleaved rounds for `highs-ds` and `highs-ipm` per instance;
- solver cutoff = 1.0 s; controller hard timeout = 3.0 s;
- thread environment = 1;
- no missing worker outcomes or worker errors;
- objective mismatch = 0 whenever both routes return comparable optimal objectives;
- selector choices are computed only from structural features and the already-frozen model;
- policy cost includes inspection cost;
- test policy / test SBS <= 0.97;
- paired bootstrap 95% upper bound of policy/SBS < 1.0 (20,000 deterministic draws);
- p90 per-instance policy/default-route cost <= 1.25;
- no harmful selected switch with >0.10 s absolute solver regret;
- at least 2 selector switches.

If **any** held-out gate fails, `performance_ranking_validated=False`, the test is considered consumed, and no M27 threshold/model change is permitted. A later selector would require a new untouched test cohort.

## Production boundary
Passing the test gate authorizes only this exact DS-vs-IPM selector for continuous LPs under compatible proof/capability gates. It does not authorize MILP/QP routing or any other backend ranking.
