# M7 Test Report

## Pytest

Final JUnit accounting on M7 source:

- total test cases represented: 139;
- passed: **136**;
- skipped: **3**;
- failures: **0**;
- errors: **0**.

The three skips are collection-level optional integration modules for:

1. public highspy;
2. public OSQP;
3. public PySCIPOpt.

They are skipped because those packages are unavailable in the current runtime; they are not counted as successes.

## Randomized / semantic regressions

- mutation/hash: **1000/1000**;
- OSQP translation contract: **1200/1200**;
- MPS parser semantic cross-check against native HiGHS reader: **200/200**;
- max MPS objective difference: `7.105427357601002e-15`;
- diagnostics planted-conflict property checks: **200/200**;
- native HiGHS IIS stress: **50/50**.

## New M7 tests

- real base backends actively probe healthy;
- unavailable optional native adapters remain visible in the candidate audit;
- a deliberately broken backend is detected as unhealthy because independent validation fails;
- backend-health report is JSON serializable;
- backend-health CLI exits successfully when no available backend is unhealthy;
- public-corpus manifest has valid source/hash/result references and all recorded cross-check invariants pass.

## Import-path trap found and controlled

During development, an initial pytest invocation resolved an older editable M6 install instead of the M7 source tree. M7 verification therefore uses an explicit source path during source tests and a separate isolated target directory during wheel tests. Release conclusions are based on those controlled runs, not the stale editable import.
