# M8 Release Check

**Version:** 0.0.9

## Source verification

- JUnit accounting: **151 total / 148 passed / 3 optional integration skips / 0 failures/errors**;
- `compileall` over source/tests/benchmarks/tools: pass;
- mutation/hash randomized checks: 1000/1000;
- OSQP translation checks: 1200/1200;
- MPS semantic cross-check: 200/200;
- diagnostics property checks: 200/200;
- native IIS stress: 50/50;
- runtime-resilience fault injection: 100 cases;
- selector stability benchmark: completed (240 solver calls);
- repeated-LP reoptimization benchmark: completed.

## Build verification

Wheel built offline from final source:

`dist/optimind_core_codename-0.0.9-py3-none-any.whl`

SHA-256:

`09f3121f8a2d22bdfb38fbff41912728dfe1d39256064ffd9fefca098faf801f`

Wheel contents explicitly include M8 runtime modules `optimind/runtime/portfolio.py` and `optimind/runtime/budgeting.py` plus the updated planner and trace schema.

## Isolated installed-wheel smoke

The final wheel was installed with `--no-deps --target` into a separate directory and imported from outside the source tree while using the base runtime's installed NumPy/SciPy dependencies.

Verified from the installed wheel:

- package version `0.0.9`;
- active health-gated solve (`require_healthy`);
- trace schema `0.3` and `planner_health_policy`;
- stale-version health report is not trusted;
- portfolio recovers after an injected backend exception;
- installed backend-health report shows distribution version and package version both `0.0.9`;
- four base backends healthy; three optional public native backends unavailable.

An earlier venv attempt could not import SciPy in this harness despite `--system-site-packages`; that failed environment setup is **not** counted as a wheel pass. The separate-target smoke above is the successful release check.

## Public-native network attempt

Exact official locked wheel URLs for `highspy`, `osqp`, and `pyscipopt` were retried. All three failed before transfer with DNS error `Could not resolve host: files.pythonhosted.org` (curl return code 6). No wheel was accepted without successful SHA-256 verification.

## Known blockers

- public `highspy 1.15.1` integration execution;
- public `OSQP 1.1.3` execution/update/warm-start;
- public `PySCIPOpt 6.2.1` execution/IIS;
- cross-solver public-corpus SBS/VBS;
- full Netlib/MIPLIB/QPLIB performance suites;
- production empirical selector.
