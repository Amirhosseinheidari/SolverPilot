# M26 Status — Cost/Regret-Aware Selector Validation

**Version:** 0.0.31  
**State:** COMPLETE / VERIFIED NEGATIVE RESULT

M26 attempted the first learned DS-vs-IPM selector only after M25 established a real public/OOD selection opportunity. The selector protocol was frozen before final test evaluation.

## Frozen protocol

- exact M25 48-instance public/OOD cohort,
- deterministic group-safe split: 20 train / 12 validation / 16 sealed test,
- obvious variants grouped (`csched`, `markshare`, `ic97/icir97`),
- train/validation artifact contains **no test costs or winner labels**,
- one low-capacity model family only: ridge regression on `log(IPM_cost / DS_cost)`,
- five structural features, standardized on train only,
- fixed 3x3 hyperparameter grid (9 candidates),
- feature/inspection cost included in policy cost,
- test evaluation forbidden unless the validation gate passes.

## Validation result

**Validation gate failed.**

Best candidate:
- ridge lambda: 10.0
- threshold: 0.0
- validation SBS: `highs-ipm`
- validation policy/SBS: **1.1886926884**
- DS choices: 10 / 12
- IPM choices: 2 / 12

Required validation policy/SBS was <= 0.98. Every one of the 9 pre-registered hyperparameter candidates was worse than SBS; the best was 1.1887 and the worst was 1.2218.

## Leakage boundary

The 16-instance test split remains sealed:
- test rows in training artifact: 0
- test names in training artifact: 0
- model reports `test_outcomes_seen_during_fit=false`
- no `m26-test-results.json` or `m26-test-evaluation.json` exists

M26 intentionally stops at validation failure. Looking at test after this failure would waste the only held-out gate and create model-selection leakage.

## Product consequence

- M25 opportunity remains validated.
- M26 selector performance ranking is **not** validated.
- `supports_performance_ranking = False`.
- LP production routing remains conservative (`scipy-highs-ds`).
- Manual IPM performance override remains rejected under M26 evidence.

The correct next research step is not to retune this ridge model against the sealed test. A future milestone must define a new cost/regret objective using only train+validation (e.g. absolute regret/selective switching), freeze it, and use the still-sealed M26 test exactly once.
