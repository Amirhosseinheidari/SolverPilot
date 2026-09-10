# M27 Status — Selective Absolute-Regret LP Routing

**Version:** 0.0.32  
**State:** COMPLETE / VERIFIED NEGATIVE HELD-OUT RESULT  
**Production consequence:** selector NOT promoted; LP performance routing remains disabled.

## Research question

M25 established a real public/OOD DS-vs-IPM opportunity. M26 showed that a relative log-cost ridge model did not capture it safely. M27 asks whether a low-capacity, abstaining selector can improve on the development SBS by switching only when the **predicted absolute wall-clock saving** is large and empirically supported.

## Frozen design

M27 was frozen before opening the original 16-instance test:

- development data: the 32 M26 train+validation rows only,
- final test: the original 16 M26 test instances,
- default development backend: `scipy-highs-ipm`,
- alternate: `scipy-highs-ds`,
- target: `default_cost - alternate_cost` in seconds,
- model: depth-2 CART regressor, minimum leaf size 8,
- features: five cheap structural features (`n`, `m`, `nnz`, density, `m/n`, log transformed),
- switch only when leaf mean saving > 5 ms, empirical one-sided 90% bootstrap lower mean > 5 ms, leaf support >= 8, and query is within observed support for tree-used features,
- every policy decision pays measured inspection cost,
- no post-test retuning is permitted.

The protocol SHA-256 is:

`2824ee0d21bf5c8e7b4bf1758b107ab569a62830032e1c11f8fb2eedee1a2ba3`

The frozen selector file SHA-256 is:

`3e095c328f401fb513a459b09caae3a6a94fa7c3641ef9cdc317f878a78a265f`

## Development result

Group-safe out-of-fold development evaluation passed its development-only gate:

- development SBS: `scipy-highs-ipm`
- SBS mean: 0.1044861 s
- policy mean: 0.1024160 s
- policy/SBS: **0.9801884**
- switches: 6 / 32
- harmful alternate switches: 0
- p90 policy/default: 1.0759

This result is **not promotion evidence** because M26 validation had already influenced the M27 research direction. All 32 rows are therefore development data.

## One-time held-out result

The artifact records a one-time 16-instance test execution after the selector and protocol were frozen. Final audit verified the frozen hashes and filesystem chronology; this is not an external append-only attestation that no unrecorded prior run ever occurred.

- exact test instances: 16
- DS/IPM rounds: 2 each
- expected outcomes: 64
- recorded outcomes: 64
- worker errors: 0
- hard timeouts: 0
- solver-level error outcomes: 2 (both DS on one instance; charged at cutoff)
- objective mismatches: 0
- test SBS: `scipy-highs-ipm`
- SBS mean: **0.1442225 s**
- M27 policy mean: **0.1453332 s**
- policy/SBS: **1.0077011**
- paired bootstrap 95%: **[0.99643, 1.02718]**
- switches: 2 / 16
- maximum harmful switch regret: 0.0154641 s
- p90 policy/default: 1.04257

The pre-registered promotion gates failed on:

- `policy/SBS <= 0.97`: FAIL
- bootstrap 95% upper bound < 1: FAIL

All correctness/accounting/tail-safety gates passed.

## Product consequence

- `selection_opportunity_validated = True`
- `performance_ranking_validated = False`
- `supports_performance_ranking = False`
- manual speed override remains rejected
- LP production routing remains conservative (`scipy-highs-ds` in the current production planner)

The M27 selector is a research artifact only.

## Irreversible test boundary

The original 16-instance M26/M27 test is now **consumed**. The final provenance audit also found that the raw held-out payload did not embed a test-time environment fingerprint, so production evidence conservatively keeps `fixed_environment=False`; this cannot change the negative conclusion, only prevent overclaiming. It must never be reused to tune or promote a future learned selector. Any future selector requires a new untouched public/OOD test cohort.

## Final audit boundary

M27 remains a negative held-out result after provenance hardening. The negative conclusion is stronger than the deployment evidence: because the raw test payload omitted explicit test-time environment and corpus-integrity fields, the production evidence translator refuses to infer them and keeps both timing provenance flags conservative. The test must not be rerun or retuned.
