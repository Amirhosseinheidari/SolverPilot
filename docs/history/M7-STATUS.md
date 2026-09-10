# M7 Status — Backend Health + Small Real Public Corpus

**Version:** 0.0.8  
**Milestone state:** completed for the explicitly listed M7 scope; public native solver package execution and large public benchmark suites remain open.

## Completed and executed

- active backend health/provenance API and installed-package CLI;
- seven built-in backend candidates surfaced, including unavailable optional adapters;
- active smoke validation for LP/MILP/convex-QP capabilities;
- four base-environment backend implementations report healthy in the release environment;
- optional `highspy`, OSQP and PySCIPOpt adapters report unavailable rather than being treated as verified;
- public-corpus audit manifest without redistributing third-party MPS inputs;
- AFIRO LP real-public benchmark evidence retained and re-audited;
- p0033 real public 0/1 MILP: published optimum reproduced and native-reader cross-check recorded;
- exmip1 real public MILP: canonical parser/runtime, independent equation formulation and native-reader cross-check recorded;
- M1/M2/M5/M6 correctness/property regressions rerun;
- M5 selector stability regression rerun;
- M3 native HiGHS reoptimization regression rerun;
- package version bumped to 0.0.8;
- console entry point `optimind-backend-health` added;
- wheel built and installed in a separate target; solve/MPS/diagnostics/health CLI smoke-tested.

## Latest release-environment health audit

| Backend | Status | Active smoke |
|---|---|---|
| highspy-native | unavailable | not executed |
| osqp-native | unavailable | not executed |
| pyscipopt-native | unavailable | not executed |
| scipy-highs-ds | healthy | LP |
| scipy-highs-ipm | healthy | LP |
| scipy-highs-bridge | healthy | LP + MILP |
| scipy-slsqp-qp-bridge | healthy | convex QP |

`available` is not treated as equivalent to `healthy`; the active probe additionally requires a validated known-objective solve.

## Real public-corpus evidence

### AFIRO

- class: LP;
- variables: 32;
- constraints: 27;
- canonical/native reader objective difference: ~5.68e-14;
- error to published reference objective: ~2.86e-9.

### p0033

- class: pure 0/1 MILP;
- variables: 33 integer;
- constraints: 16;
- published optimum: 3089;
- canonical result: 3089.0000000000077;
- native-reader result: 3089.0000000000077;
- canonical/native difference: 0;
- error to published optimum: ~7.73e-12.

### exmip1

- class: MILP;
- variables: 8;
- constraints: 5;
- integer variables: 2;
- canonical objective: 3.236842105263158;
- independently transcribed equation model: same objective;
- native HiGHS MPS reader: same objective;
- recorded pairwise objective differences: 0.

This is deliberately described as a small public smoke corpus, not as MIPLIB/Netlib-wide performance evidence.

## Latest selector research regression

- instances: 24;
- solver calls: 240;
- DS median wins: 12;
- IPM median wins: 12;
- stable winner (>=4/5 rounds): 23/24;
- policy/SBS cost ratio: 0.9222437375;
- bootstrap 95% interval: [0.8761913719, 0.9715794075];
- SBS→VBS gap closure: 0.8701819082;
- max objective difference: 0.

No empirical policy is enabled in the production planner. Evidence level remains `capability_only`.

## Latest repeated-LP native reuse regression

Median stateful/cold end-to-end wall ratios:

- small: 0.6045459687 (bootstrap 95% CI [0.5675347384, 0.6693222227]);
- medium: 0.4090358088 (CI [0.3894656160, 0.4328524812]);
- large: 0.3312905255 (CI [0.3133859026, 0.3679358233]).

All paired objectives agree and all expected post-initial reuse steps were backend-confirmed as applied.

## Not completed

- public highspy 1.15.1 execution in this runtime;
- public OSQP 1.1.3 execution in this runtime;
- public PySCIPOpt 6.2.1 execution in this runtime;
- full Netlib corpus;
- MIPLIB 2017 benchmark;
- QPLIB compatible subset;
- public cross-solver SBS/VBS;
- ML selector;
- public package name/legal clearance.

The external blocker for the three optional native packages remains runtime package-download/DNS/egress access, not a test pass.
