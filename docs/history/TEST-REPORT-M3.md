# Test Report — M3

**Date:** 2026-08-31  
**Package version:** 0.0.4

## Full suite

```bash
PYTHONPATH=src python -m pytest -o addopts='' -ra
```

Result:

```text
94 passed, 3 skipped
```

Skipped by design because the actual optional public packages are absent:

```text
tests/test_highspy_native_integration.py
tests/test_osqp_native_integration.py
tests/test_pyscipopt_native_integration.py
```

A skipped test is not counted as success.

## New M3 native-development tests

`tests/test_scipy_vendored_highs_dev.py` executes the real native HiGHS binary bundled privately inside SciPy and verifies:

- continuous LP optimal solve;
- same-structure explicit basis reuse;
- structural-change cold invalidation;
- MILP previous-solution injection;
- convex QP solve;
- Session propagation of backend-confirmed reuse;
- development backend is absent from default registry;
- native IIS extraction;
- per-run HiGHS runtime is stored as a delta, not mistaken for the cumulative timer.

Additional native compatibility tests pass the public-highspy adapter's LP/QP model structures into the SciPy-vendored HiGHS binary and verify they are accepted and solved.

## Property checks

```text
M1 mutation/hash:      1000 / 1000 passed
M2 OSQP translation:  1200 / 1200 passed
```

## M3 benchmark integrity checks

For 27 repeated-LP sequence measurements:

- stateful and cold objectives are cross-checked per sequence step;
- expected post-initial reuse count is 9/9 in every run;
- structural invariants are fixed by construction;
- one-time solver initialization is warmed outside timing;
- cold/stateful execution order is alternated;
- wall time and native solver-run delta are recorded separately;
- median ratio uncertainty is reported by deterministic bootstrap.

## Known gaps

No test in this environment can yet prove compatibility with the exact public packages:

- highspy 1.15.1;
- OSQP 1.1.3;
- PySCIPOpt 6.2.1.

The SciPy private HiGHS verification is deliberately labeled development-only and targets HiGHS 1.8.0.
