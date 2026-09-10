# M11 Test Report — 0.0.12

## Final pytest
JUnit artifact: `benchmarks/results/m11-pytest-junit.xml`

- collected: 204
- passed: 201
- skipped: 3
- failures: 0
- errors: 0

The three skipped modules are optional dedicated public native integrations whose packages are unavailable in this runtime: highspy, OSQP and PySCIPOpt.

## New M11 contract coverage
M11 tests cover:
- built-in MIPLIB/QPLIB dataset specs and JSON round-trip;
- local acquisition with SHA-256 verification;
- bad-hash rejection;
- archive path-traversal rejection;
- dataset re-verification and missing-instance detection;
- QPLIB extraction-count verification when no manifest exists;
- rectangular benchmark matrices;
- explicit missing-instance rows;
- deterministic disjoint sharding;
- resume semantics;
- protocol identity changes after config/seed changes;
- run identity changes when instance bytes change without filename change;
- recovery from a truncated final JSONL record;
- hard failure for corruption in the middle of JSONL;
- mixed-protocol summary rejection;
- duplicate run-ID rejection;
- process-isolated success;
- controller hard-timeout recording;
- `@auto` benchmark execution and policy metrics;
- exclusion of `@auto` from SBS/performance-profile solver candidates;
- solver-only vs deployable cost-field semantics;
- deterministic paired-bootstrap policy/SBS intervals;
- external policy map exact-coverage and overhead semantics;
- SLURM array generation;
- result bundle SHA manifest;
- CLI doctor/list/acquire/verify/run/summarize/evaluate-policy/emit-slurm/bundle surfaces.

## Numeric regression scripts re-run on M11 source
- `m1_property_checks.py`: 1000/1000.
- `m2_osqp_translation_checks.py`: 1200/1200.
- `m5_mps_parser_crosscheck.py`: 200/200, max objective diff 7.105427357601002e-15.
- `m6_diagnostics_property_checks.py`: 200/200.
- `m6_native_iis_stress.py`: 50/50.
- `m10_cross_solver_qp.py`: 48 instances, 576 direct solver calls, zero failures.
- M10 binary-MILP exact-bruteforce portion: 100/100 instances, zero failures across all three paths.
- `m10_osqp_stateful_qp.py`: all expected reuse counts 14/14 for every sequence repetition.
- `m3_highs_reoptimization.py`: all warm/cold objectives agree and every expected reuse step applied.
- `m5_lp_selector_stability.py`: completed 240 calls; 23/24 instances stable in >=4/5 rounds.

## Public p0033 regression note
The full historical `m10_cross_solver_milp.py` script could not execute its final `p0033.mps` section because that third-party file was intentionally not redistributed in the M10/M11 archive. The failure was `FileNotFoundError`, not a solver failure. The 100-instance exact-bruteforce section was re-run separately and saved as `benchmarks/results/m11-m10-milp-synthetic-regression.json`.
