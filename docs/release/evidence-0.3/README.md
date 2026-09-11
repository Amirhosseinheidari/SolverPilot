# Local 0.3 qualification evidence

Measured against the installed 0.3 wheel on Windows 11 / Python 3.12.14.
The wheel path is recorded in `performance-probes.json`; the full local wheel
regression run passed **1191 tests**, with **30 prerequisite/platform skips**.
Combined statement/branch coverage: **81.03%**; statements: **85.09%**;
branches: **69.91%**. The common API mypy contract and Ruff correctness checks passed.
GitHub qualification is a separate gate and is not implied by these local results.

## Performance

The benchmark uses one warmup and three measured rounds, randomized method order,
the same tiny LP, backend and seed. Compilation uses three repetitions per size.
OMP/MKL/OpenBLAS environment limits are set to 1; this is not proof that every
native solver thread pool is capped. Memory numbers measure Python allocations,
not native RSS. No full test suite ran concurrently with timed benchmark samples.

| Measurement | 0.2 local audit | 0.3 installed wheel |
|---|---:|---:|
| Dense semantic model + first compile, 512 variables / 256 rows, 2% density | 500.26 ms | 66.32 ms |
| New sparse semantic path, same data | unavailable | 8.37 ms |
| Twelve small LPs, new four-worker batch including startup | 3012.61 ms | 818.08 ms |
| Twelve small LPs, already-running four-worker executor | unavailable | 12.55 ms |
| Twelve LPs, direct serial execution | 33.59 ms | 30.85 ms |
| Already-cancelled generator inputs consumed | 20000 | 0 |

Cold and steady-state times are different scopes. Tiny one-off workloads still
favor serial execution over starting solver processes. The new explicit sequential
batch API took 38.08 ms here. These are synthetic examples on one host, not general
speed claims or comparisons with competing modeling libraries.

## Audit-to-regression mapping

| Finding | Implementation | Regression evidence |
|---|---|---|
| F1–F3 certificate false positives | exact sparse certificate products, domain and strong-convexity corrections | `test_03_audit_regressions`, `test_03_numeric_boundaries` |
| F4 stale OSQP settings | complete optional defaults on update | `test_03_audit_regressions` |
| F5 budget destroys reuse | scoped configuration on owned locked backend | `test_03_runtime` |
| F6 wrong-model conic result mapping | data identity on conic/NLP/MINLP result paths | `test_03_audit_regressions`, `test_03_analysis` |
| F7 mutable validated NLP result | independent read-only arrays and frozen statistics | `test_03_audit_regressions`, `test_03_runtime` |
| F8 dense symbolic coefficients | CSR affine blocks and cached unchanged dependencies | `test_03_modeling`, `test_03_edge_cases` |
| F9 eager/per-job process batches | streaming submission, restartable persistent processes, sequential mode | `test_03_runtime`, `test_03_edge_cases`, benchmark |
| F10 fragmented controls/discovery/results | common options/summary and catalog, explicit unsupported controls | `test_03_analysis`, `test_03_edge_cases` |
| F11 incomplete run recording | backend configuration, budget, identity, versioned JSON replay | `test_03_runtime`, `test_03_edge_cases` |
| F12 missing semantic stress tests | new scale, permutation, derivative, atom, scenario and process tests; CI coverage/type gates | full suite and installed-wheel qualification |

The new API deliberately has documented boundaries: no general derivative at
degenerate optima, no universal in-process cancellation/memory guarantee, no
independent conic/global NLP proof, no automatic learned solver selector/GPU claim.
See the [0.3 guide](../SOLVERPILOT-PUBLIC-DOCS-0.3.md).
