# Release Check — M24

**Package:** optimind-core-codename 0.0.29  
**Milestone:** M24 — Public/OOD Comparative Selector Falsification

## Passed

- version bump to 0.0.29: PASS
- M24 benchmark run 1: PASS as an executed comparative experiment
- unchanged M24 benchmark run 2: PASS
- identical cohort across runs: PASS
- identical winner pattern across runs: PASS
- objective agreement: PASS / 0 mismatches
- evidence-contract hardening tests: PASS
- full regression: PASS / 233 collected / 230 passed / 3 skipped / 0 failed / 0 errors

## Promotion result

LP synthetic selector promotion: **REJECTED BY EVIDENCE**.

This is an intended valid release result. M24 does not need a positive speedup to pass; it needs a reproducible, fail-closed decision about whether production routing is justified.

## Production behavior after M24

- LP baseline remains `scipy-highs-ds`.
- Failed comparative evidence cannot authorize manual or automatic speed routing.
- No MIP/QP performance superiority claim is introduced by M24.
