# Test Report M8

- JUnit total tests: **151**
- passed: **148**
- skipped: **3**
- failures: **0**
- errors: **0**

The three skips are the optional public `highspy`, `osqp`, and `pyscipopt` integration modules in the current environment. Skips are not counted as passes.

Additional executed checks:

- mutation/hash property checks: 1000/1000;
- OSQP translation checks: 1200/1200;
- MPS semantic cross-check: 200/200;
- diagnostics property checks: 200/200;
- native IIS stress: 50/50;
- runtime-resilience fault injection: 100 cases;
- selector stability: 240 solver calls;
- repeated-LP reoptimization regression: completed.

Trace schema is now `0.3`.
