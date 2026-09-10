# BENCHMARK REPORT M28

## Protocol

Primary benchmark: 48 fresh MIPLIB-derived continuous LP relaxations, 3 rounds, process-isolated workers, 1 solver thread, 1-second solver budget, 3-second controller hard timeout. Route order was deterministically interleaved. Exact protocol: `docs/research/M28-NATIVE-CHOOSE-BASELINE-PROTOCOL.md`.

## Accounting

- Expected native outcomes: **432** (`48 × 3 routes × 3 rounds`).
- Recorded native outcomes: **432**.
- Worker/controller errors: **0**.
- Objective mismatch instances: **0**.
- Corpus integrity: **True**.
- Same fixed-host/runner provenance: **True**.

## Cost results

- `choose`: 0.189512244 s mean.
- `simplex`: 0.191102338 s mean.
- `ipm`: 0.261087382 s mean.
- Forced SBS: `simplex` at 0.191102338 s.
- Forced VBS: 0.171458574 s.
- SBS→VBS relative opportunity: **10.279%**.
- `choose` relative gain vs SBS: **0.832%**, but paired bootstrap includes 1: [0.974413, 1.006492].
- `choose` closes only **8.095%** of the available forced-route gap.

Forced-route per-instance winners (0.5% tie band): simplex 36, IPM 8, ties 4.

## Algorithm-use audit

`choose` iteration diagnostics:

```json
{
  "presolve_or_undetermined": 3,
  "simplex": 141
}
```

No nontrivial `choose` run recorded IPM iterations. Therefore M28 does not call this an instance-wise DS/IPM selector.

## Invalidity audit

Campaign invalid counts:

```json
{
  "choose": 12,
  "ipm": 15,
  "simplex": 12
}
```

These are deliberately assigned cutoff cost. Post-hoc diagnosis is in `benchmarks/results/m28/m28-invalidity-posthoc.json`; the frozen campaign was not re-tuned or rerun.

## Claim boundary

This is a comparison of the verified **bundled HiGHS 1.10.0** core only. It is not a benchmark of current public highspy 1.15.x. SciPy 1.17.0 semantics are recorded separately and do not enter the primary cost matrix.
