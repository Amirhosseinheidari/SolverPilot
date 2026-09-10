# M28 STATUS — Native-Choose Baseline Audit

Version: **0.0.33**

Classification: **VERIFIED NEGATIVE BASELINE RESULT — NO PERFORMANCE ROUTING AUTHORITY**

## Question

Does HiGHS' own `choose`/default route already close enough of the DS-vs-IPM portfolio opportunity that an external OptiMind selector is unnecessary?

## Fresh cohort

- Source: official MIPLIB 2017 benchmark-v2 archive locked in M22.
- 48 continuous LP relaxations.
- 0 overlap with the 48-instance M25 cohort (and therefore 0 overlap with M26/M27 consumed test data).
- Cohort signature: `800c50ea8bb68c87283902d34fb6e0677a730f2fec7ca423b56643ff11ab3327`.
- Corpus SHA-256: `c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa`.
- Selection was outcome-independent and hash-ordered.

## Same-core comparison

All timed primary routes used **the same bundled HiGHS 1.10.0 C API build**:

- `choose`
- `simplex`
- `ipm`

This avoids comparing solver-core versions.

## Result

| metric | value |
|---|---:|
| choose mean | 0.189512 s |
| simplex mean | 0.191102 s |
| IPM mean | 0.261087 s |
| forced SBS | `simplex` |
| forced SBS→VBS gap | 10.28% |
| choose/SBS | 0.991679 |
| choose/SBS bootstrap 95% | [0.974413, 1.006492] |
| choose VBS-gap closure | 8.09% |

`choose` diagnostics across 144 runs:

- simplex: **141**
- IPM: **0**
- presolve/undetermined: **3**

So this verified HiGHS 1.10 build did not use IPM under `choose` on any nontrivial M28 run.

## Gate

`native_choose_closes_portfolio_opportunity = false`

Failed gates include:

- IPM terminal rate >= 90%: **False** (actual 0.8958)
- zero independent invalidity: **False**
- choose closes >=80% of the forced SBS→VBS gap: **False**

## Independent-validation finding

Invalid candidates were not promoted to feasible/optimal outcomes. Post-hoc diagnosis found that most were time-limit candidates with large primal violations; one forced-IPM `optimal` candidate had max constraint violation around `1.9e-7`, slightly above OptiMind's independent `1e-7` feasibility tolerance. The campaign was not rerun with looser tolerances.

## Production decision

- Keep automatic LP performance routing **OFF**.
- Keep conservative public baseline behavior unchanged.
- Do not represent SciPy `method="highs"` as a distinct auto-selector in SciPy 1.17.0; local implementation maps it to `solver=None`, whose private HiGHS wrapper documents simplex fallback.
- M28 confirms a real DS/IPM opportunity remains, but neither native `choose` nor the failed M27 selector captures it safely.
