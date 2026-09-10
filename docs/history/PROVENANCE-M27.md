# Provenance — M27

M27 is derived from the exact M25 48-instance public/OOD MIPLIB-derived LP-relaxation cohort and the M26 frozen split.

Primary inputs:
- `benchmarks/results/m25-cohort.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-run1.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-run2.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-stability.json`
- `benchmarks/results/m26/m26-frozen-split.json`
- `benchmarks/results/m26/m26-trainval-only.json`

M27 frozen pre-test artifacts:
- `docs/research/M27-SELECTIVE-ABSOLUTE-REGRET-PROTOCOL.md`
- `benchmarks/results/m27/m27-protocol.sha256`
- `benchmarks/results/m27/m27-frozen-selector.json`
- `benchmarks/results/m27/m27-frozen-selector.file.sha256`
- `benchmarks/results/m27/m27-development-oof-audit.json`

One-time held-out artifact:
- `benchmarks/results/m27/m27-one-time-heldout-test.json`

The held-out artifact verifies the exact protocol/model/split hashes before running. It refuses overwrite, so the M27 one-time test cannot silently be replaced by a later run.

## Test-consumption boundary

M26 never opened the final 16-instance test. M27 opened it once after freeze. Those 16 instances are now development-visible history and **must not be used as future held-out promotion evidence**.

## Runtime implementation

M27 adds a pure-Python runtime evaluator in `src/optimind/plan/selective_lp.py`. The serialized tree does not require scikit-learn at runtime. Training used scikit-learn 1.8.0, but production ranking remains disabled because the held-out gates failed.
