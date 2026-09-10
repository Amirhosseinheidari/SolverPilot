# M7 Benchmark Report

M7 adds a small real public corpus and reruns the existing selection/reoptimization evidence. It does not claim a full public-suite benchmark.

## 1. Real public corpus

Third-party MPS bytes are not redistributed. Exact source URLs and locally benchmarked SHA-256 hashes are recorded in `benchmarks/public-corpus-m7.json`.

### AFIRO — LP

- canonical/parser solve and direct native HiGHS file-reader solve agree to ~`5.68e-14` in objective;
- both reproduce the published reference objective to ~`2.86e-9`.

### p0033 — 0/1 MILP

Public file metadata states 33 integer columns, 16 rows and best solution `3089 (opt)`.

Recorded M7 result:

- canonical OptiMind objective: `3089.0000000000077`;
- native HiGHS direct-reader objective: `3089.0000000000077`;
- canonical/native difference: `0`;
- absolute error to published optimum: `7.73e-12`;
- canonical solution independently validates.

### exmip1 — MILP

Recorded M7 result:

- canonical parser/runtime objective: `3.236842105263158`;
- independently transcribed equation model objective: `3.236842105263158`;
- native HiGHS direct MPS reader objective: `3.236842105263158`;
- recorded pairwise differences: `0`;
- canonical solution independently validates.

This provides real external smoke evidence for LP and MILP ingestion/solution, but three instances are not enough for comparative performance claims.

## 2. LP algorithm-selection stability regression

Method remains the M5 research-only design: 24 structured synthetic LPs, 2 HiGHS algorithms, 5 interleaved timing rounds per solver/instance = 240 calls.

Latest M7 run:

- DS median wins: 12;
- IPM median wins: 12;
- 23/24 instances have the same winner in at least 4/5 rounds;
- SBS: `scipy-highs-ds`;
- policy/SBS aggregate cost ratio: `0.9222437375`;
- bootstrap 95% interval (10,000 draws): `[0.8761913719, 0.9715794075]`;
- SBS→VBS gap closure: `0.8701819082`;
- max DS/IPM objective difference: `0`.

Interpretation: the controlled synthetic portfolio still shows stable complementary performance. This evidence is **not** promoted into the production planner because it is not cross-solver and not validated on a substantial public corpus.

## 3. Native repeated-LP reoptimization regression

Latest M7 run, 9 sequence runs per size:

| Size | Median stateful/cold wall ratio | Bootstrap 95% CI | Approx speedup | Median iteration ratio |
|---|---:|---:|---:|---:|
| small | 0.6045 | [0.5675, 0.6693] | 1.65x | 0.1450 |
| medium | 0.4090 | [0.3895, 0.4329] | 2.44x | 0.1720 |
| large | 0.3313 | [0.3134, 0.3679] | 3.02x | 0.1647 |

All paired objectives agree. All expected reuse steps were reported by the backend as applied.

This remains a controlled repeated-LP workload over the development-only native HiGHS verifier, not a universal performance claim and not yet an exact-public-highspy result.

## 4. Backend-health audit

M7 actively probes known-objective canonical smoke problems rather than equating import availability with health.

Healthy in the release environment:

- scipy-highs-ds: LP;
- scipy-highs-ipm: LP;
- scipy-highs-bridge: LP + MILP;
- scipy-slsqp-qp-bridge: convex QP.

Unavailable and therefore unverified:

- highspy-native;
- osqp-native;
- pyscipopt-native.

No available backend was reported unhealthy in the release audit.
