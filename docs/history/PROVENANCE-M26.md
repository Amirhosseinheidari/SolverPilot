# Provenance — M26

M26 is derived from M25's exact public/OOD 48-instance MIPLIB-2017 LP-relaxation cohort and two unchanged M25 timing runs.

Inputs:
- `benchmarks/results/m25-cohort.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-run1.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-run2.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-stability.json`

Frozen M26 artifacts:
- `benchmarks/results/m26/m26-frozen-split.json`
- `benchmarks/results/m26/m26-frozen-split-map.json`
- `benchmarks/results/m26/m26-trainval-only.json`
- `benchmarks/results/m26/m26-frozen-model.json`
- `benchmarks/results/m26/m26-validation-diagnostics.json`

The M26 final held-out test contains 16 named instances in the split artifact, but their M25 timing outcomes were not copied into the training artifact and were not evaluated by the M26 training/validation scripts after the validation gate failed.
