# M24 Status — Public/OOD Comparative Selector Falsification

**Version:** 0.0.29  
**State:** COMPLETE / VERIFIED

M24 asked the question M23 deliberately left open: does the previously promising synthetic LP selector generalize to public, out-of-development-distribution structure strongly enough to authorize production performance routing?

## Result

**No.** The pre-registered M4/M5 rule (`IPM if m/n >= 1.5`) failed the M24 promotion gate on a deterministic resource-bounded cohort of 24 continuous LP relaxations derived from official MIPLIB 2017 benchmark-v2 models.

Two unchanged full timing runs produced the same cohort and the same winner pattern:

- dual simplex wins: **22 / 24**
- IPM wins: **1 / 24**
- ties: **1 / 24**
- objective mismatches: **0**
- rule IPM choices: 3 instances
- actual IPM-winning instance: 1 different instance
- selector promoted: **false** in both runs

Run 1 policy/SBS ratio: **1.01951**, bootstrap 95% interval **[1.00667, 1.07747]**.  
Run 2 policy/SBS ratio: **1.01965**, bootstrap 95% interval **[1.00679, 1.07841]**.

The old selector therefore made deployment worse on this public/OOD cohort after paying inspection cost.

## Product consequence

M24 **does not enable LP performance auto-routing**. Production keeps the conservative `scipy-highs-ds` LP baseline.

M24 also hardens the evidence contract: `COMPARATIVE_HELDOUT` evidence is no longer sufficient by itself to authorize performance overrides. Production ranking now additionally requires:

- public/OOD evidence,
- a pre-registered policy,
- fixed-environment timing,
- feature-cost accounting,
- at least two compared backends,
- and an explicitly successful promotion gate.

Negative comparative evidence is retained as useful evidence but cannot authorize routing.

## Verification

- full regression: **233 tests / 230 passed / 3 skipped / 0 failed / 0 errors**
- two public/OOD timing runs: reproducible cohort and winner pattern
- official MIPLIB archive integrity inherited and rechecked against M22 transfer-integrity evidence
- production override with failed M24 evidence: rejected as designed
