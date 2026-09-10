# Provenance — M25

**Version:** 0.0.30

## Source lineage

M25 is derived from the verified M24 source tree. It retains M22 official MIPLIB corpus-integrity evidence and M24 public/OOD negative-selector evidence.

## New M25 evidence

- `benchmarks/results/m25-cohort.json`
- `benchmarks/results/m25-opportunity-shard0.json` ... `shard3.json` (run 1)
- `benchmarks/results/m25-run2-shard0.json` ... `shard3.json` (run 2)
- `benchmarks/results/m25-public-ood-lp-opportunity-run1.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-run2.json`
- `benchmarks/results/m25-public-ood-lp-opportunity-stability.json`

## New code

- `benchmarks/m25_public_ood_lp_opportunity.py` — frozen monolithic protocol/preregistration reference
- `benchmarks/m25_prepare_cohort.py` — deterministic cohort materialization
- `benchmarks/m25_solve_worker.py` — clean process-isolated solve worker
- `benchmarks/m25_run_shard.py` — deterministic shard executor
- `benchmarks/m25_merge_opportunity.py` — fail-closed complete-shard merger and opportunity gate
- `src/optimind/plan/production.py` — adds `selection_opportunity_validated` to the production evidence contract and the M25 evidence parser

## Execution adaptation boundary

The original monolithic 48-instance run exceeded the outer execution window. Acceptance criteria, cohort rules, cutoffs, rounds, and opportunity thresholds were not changed. Execution was split into deterministic shards and solve calls were moved to clean process-isolated workers. The final merger requires all four shard indices, 48 unique rows, identical cohort signature, and exact cohort membership before computing metrics.
