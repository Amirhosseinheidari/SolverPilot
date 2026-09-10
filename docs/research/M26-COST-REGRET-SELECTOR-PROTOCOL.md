# M26 Cost/Regret-Aware LP Selector Protocol (Frozen Before Test Evaluation)

## Purpose
Build the first learned DS-vs-IPM selector only because M25 validated a non-trivial public/OOD SBS->VBS opportunity. The selector is not allowed into production merely because a model can be fit.

## Leakage controls
- Cohort is exactly the 48-instance M25 public/OOD cohort; no instance is added/removed after outcomes.
- Split is deterministic from SHA-256(group-id), not cost/winner labels.
- Obvious variants are group-locked: `csched007/008`, `markshare*`, and `ic97/icir97`.
- Target sizes: train=20, validation=12, test=16.
- Training artifact contains no test costs or winner labels.
- Hyperparameters are selected only on validation.
- Model is serialized and SHA-256 frozen before test outcomes are evaluated.

## Model family
One low-capacity ridge model predicts `log(IPM_cost / DS_cost)` from 5 structural features:
`log1p(n)`, `log1p(m)`, `log1p(nnz)`, `log(density)`, `log1p(m/n)`.
Standardization is fit on train only.

Hyperparameter grid is fixed:
- ridge lambda in {0.1, 1.0, 10.0}
- selection threshold in {-0.05, 0.0, 0.05}

IPM is selected when predicted log-ratio < threshold; otherwise DS is selected.
Training targets/costs use the robust median across M25 run1+run2 (each itself has two process-isolated rounds).

## Cost accounting
Selector policy cost = measured `inspect_problem` cost + measured selected solver cost.
SBS has no selector feature cost. Timeouts remain cutoff-valued outcomes.

## Validation gate (required before any promotion can be considered)
- validation policy/SBS <= 0.98
- at least 2 validation instances select each route (avoid degenerate apparent selector)
- objective mismatches in source evidence = 0

## Final held-out deployment gate
Evaluated after the model artifact is frozen:
- test instances = 16 exactly, split integrity/group leakage = PASS
- fresh M26 test run: policy/SBS <= 0.97
- paired bootstrap 95% upper bound of policy/SBS < 1.0 (20,000 draws; deterministic seed)
- p90 per-instance policy cost / test-SBS-route cost <= 1.25
- fresh objective mismatches = 0 and no missing worker outcomes
- retrospective stability audit: the same frozen model has policy/SBS < 1.0 on M25 run1 test rows and M25 run2 test rows independently
- feature cost included in all selector policy costs

If any gate fails, `performance_ranking_validated=False` and production LP routing stays conservative. No post-test threshold, feature, split, or model-family changes are allowed within M26.
