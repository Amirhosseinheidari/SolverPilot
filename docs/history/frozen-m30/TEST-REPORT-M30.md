# Test Report — M30

## Final regression

`PYTHONPATH=src pytest` with JUnit output:

- collected: 283
- passed: 280
- skipped: 3
- failures: 0
- errors: 0

The three skips correspond to optional public native integrations not installed in the active verification runtime. They are preserved as skips rather than reclassified as passes.

## Property / numerical campaigns

| Campaign | Result |
|---|---:|
| Mutation/hash invariants | 1000 / 1000 |
| OSQP translation/same-sparsity invariants | 1200 / 1200 |
| Infeasibility diagnostics | 200 / 200 |
| Direct HiGHS C-API exact MILP cross-check | 100 instances / 300 calls / 0 failures |
| Direct OSQP C-API convex-QP cross-check | 96 instances / 192 calls / 0 failures |

Direct HiGHS MILP maximum objective absolute error against the independent references was approximately `1.28e-13`. Direct OSQP maximum objective absolute error was approximately `1.78e-15`.

## Reoptimization correctness rerun

For all controlled LP and QP sizes in the M30 rerun:

- stateful and cold objectives agreed,
- expected reuse was observed.

The measured stateful/cold timing ratios are retained in raw evidence, but M30 treats them as workload-local diagnostics rather than a new universal speed claim.

## Distribution smoke

Final wheel isolated-target smoke:

- version: `0.0.35`
- import outside source tree: yes
- public API count: 78
- LP: validated optimal
- MILP: validated optimal
- QP: validated feasible (SLSQP route has no optimality-certificate claim)
- production LP backend: `scipy-highs-ds`
- learned performance auto-routing: off
- stable unknown-backend exception: pass
- stable unsupported-budget exception: pass
- experimental namespace: available
- backend-health CLI: executed
- benchmark CLI help: executed

Final sdist was unpacked, rebuilt into a wheel offline, installed outside the source tree, and solved a validated LP successfully.
