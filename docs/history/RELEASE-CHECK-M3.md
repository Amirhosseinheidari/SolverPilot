# Release Check — M3 Research Artifact

**Date:** 2026-08-31  
**Package version:** 0.0.4  
**Release type:** research/development artifact, not public production release

## Verification executed on the exact source tree before packaging

- `python -m compileall -q src tests benchmarks` — passed
- `python -m pip install -e . --no-build-isolation --no-deps` — passed
- import/version smoke — `optimind.__version__ == 0.0.4`
- full pytest — **94 passed, 3 skipped**
- mutation/hash property checks — **1000/1000 passed**
- OSQP translation property checks — **1200/1200 passed**
- repeated-LP HiGHS development benchmark — **27 paired sequence runs**

The three skipped tests are exact-public-package integration tests for `highspy`, `osqp`, and `pyscipopt`; they are skipped because those packages cannot be installed in this runtime. They are not counted as passes.

## Latest repeated-LP verification

| Size | Median stateful/cold wall ratio | 95% bootstrap CI | Equivalent median speedup | Median simplex iteration ratio |
|---|---:|---:|---:|---:|
| 80×200 | 0.630 | [0.601, 0.688] | 1.59× | 0.145 |
| 150×400 | 0.421 | [0.414, 0.437] | 2.37× | 0.172 |
| 250×700 | 0.346 | [0.330, 0.365] | 2.89× | 0.165 |

All warm/cold objective checks agree and every expected post-initial basis-reuse step is reported as applied.

## Boundaries of the evidence

The native stateful verification uses SciPy 1.17.0's private vendored HiGHS 1.8.0 binding. It is deliberately development-only and absent from the default backend registry. It does **not** prove exact behavior of public `highspy 1.15.1`.

No MIPLIB/QPLIB public-corpus result, native OSQP result, native PySCIPOpt result, Benchopt run, or generally superior solver-selector claim is included in M3.
