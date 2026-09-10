# M12 Test Report

The full test inventory was executed in non-overlapping batches because the process-isolated benchmark tests make a single pytest invocation exceed this environment's controller window. The split does **not** remove tests; JUnit aggregation verifies **209 unique test cases with no duplicates**.

| Gate | Tests | Passed | Skipped | Failed/Error |
|---|---:|---:|---:|---:|
| benchmark_harness | 11 | 11 | 0 | 0 |
| benchmark_misc | 26 | 26 | 0 | 0 |
| casadi | 9 | 9 | 0 | 0 |
| core | 163 | 160 | 3 | 0 |
| **Total** | **209** | **206** | **3** | **0** |

The three skips are the optional public integrations `highspy`, `osqp`, and `PySCIPOpt`, which are not installed in the runtime.

Additional non-pytest checks:
- mutation/hash 1000/1000;
- OSQP translation 1200/1200;
- MPS parser/native-HiGHS cross-check 200/200;
- diagnostic randomized checks 200/200;
- native HiGHS IIS stress 50/50.

Raw aggregate JUnit: `benchmarks/results/m12-pytest-junit.xml`.
