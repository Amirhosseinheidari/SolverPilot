# TEST REPORT — M9

**Version:** 0.0.10  
**Final source JUnit:** `benchmarks/results/m9-pytest-junit.xml`

## Pytest

- collected: **160**
- passed: **157**
- skipped: **3**
- failures: **0**
- errors: **0**

The three skips remain the optional public integrations for `highspy`, `osqp`, and `pyscipopt` in this environment. `nlopt` is installed and its tests execute rather than skip.

## Numeric/property regressions

- mutation/hash: 1000/1000
- OSQP translation: 1200/1200
- MPS parser/native semantic cross-check: 200/200
- diagnostics: 200/200
- native IIS stress: 50/50
- manifest/reference smoke: 7/7 reference matches
- health/fallback fault injection: 100/100 healthy routing; 100/100 recovery

## New M9 coverage

`tests/test_nlopt_native.py` covers:
- LP minimization and maximization;
- convex QP equality KKT optimum;
- active inequality KKT optimum;
- MILP rejection;
- no false infeasibility proof;
- default planner does not displace native HiGHS LP with NLopt.

A new QP portfolio regression test verifies exactly one attempt per requested backend. This test was added after release smoke found and fixed the duplicate-attempt bug.
