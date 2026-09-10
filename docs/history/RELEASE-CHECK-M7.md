# M7 Release Check

**Version:** 0.0.8

## Source verification

- source-path pytest: pass;
- JUnit accounting: 136 passed, 3 optional integration skips, 0 failures/errors;
- `compileall` over source/tests/benchmarks/tools: pass;
- mutation/hash checks: 1000/1000;
- OSQP translation checks: 1200/1200;
- MPS semantic cross-check: 200/200;
- diagnostics property checks: 200/200;
- native IIS stress: 50/50;
- selector stability benchmark: completed (240 solver calls);
- repeated-LP reoptimization benchmark: completed.

## Build verification

Wheel built offline from final source:

`dist/optimind_core_codename-0.0.8-py3-none-any.whl`

SHA-256:

`223a18bc4b3a77aea9ec7484ec1aae9df08a8aba5963aa8c0e41f9e52dab3e77`

## Isolated wheel smoke

Wheel installed with `--no-deps --target` into a separate directory and tested from outside the source tree.

Verified from installed wheel:

- package version `0.0.8`;
- validated LP solve with expected objective;
- MPS parse + solve with expected objective;
- infeasible solve with `diagnose_infeasible=True` and confirmed diagnostic report;
- backend health report generation;
- installed `optimind-backend-health` console command;
- no backend classified `unhealthy` in the release environment.

## Known release blockers

Public optional integrations are still unexecuted because the runtime cannot retrieve/install the official native wheels. Current release therefore remains a research core, not a public production release.

Blocked/unverified here:

- `highspy 1.15.1` public package integration;
- `osqp 1.1.3` public package integration;
- `PySCIPOpt 6.2.1` public package integration;
- Benchopt public harness execution;
- MIPLIB/QPLIB/full Netlib performance suites.
