# M33 Final Closure — External CI Execution & Publication Decision

**Version:** `0.0.36rc3`  
**Milestone:** **FINAL VERIFIED / CLOSED**  
**Public RC:** **NOT AUTHORIZED**  
**Public 1.0:** **NOT AUTHORIZED**

M33 closes the internal engineering work required to define and harden the external compatibility/publication gate. It does not convert unexecuted GitHub-hosted compatibility cells, unavailable vulnerability/Twine tooling, or unresolved owner metadata into passes.

## Frozen policy

- intended public Python support: CPython 3.12 / 3.13 / 3.14;
- intended public OS support: Linux / macOS / Windows;
- minimum runtime floors: NumPy >=2.2, SciPy >=1.15;
- learned LP performance routing: OFF;
- exact-artifact build-once/test-many compatibility contract;
- full-SHA GitHub Action pinning;
- OIDC / Trusted Publishing release path, no long-lived PyPI token;
- artifact attestation generation + verification required before public RC;
- runtime vulnerability audit / CycloneDX evidence required before public RC.

## Recertified local evidence

- release-contract suite: PASS after generated metadata cleanup;
- full regression: 316 collected / 312 passed / 4 skipped / 0 failures / 0 errors;
- exact rc3 wheel LP/MILP/QP/production smoke outside source tree: PASS;
- NLopt optional backend on Linux/Python 3.13: PASS;
- CasADi 3.7.2 verification bridge on Linux/Python 3.13: PASS;
- sdist -> wheel -> isolated-target smoke: PASS;
- wheel reproducibility with fixed SOURCE_DATE_EPOCH: byte-identical PASS;
- sdist same-source comparison: semantic content identical, bytes not identical;
- source tree generated `egg-info`/build/cache pollution removed before final archive.

## External gates intentionally still closed

- 9/9 core GitHub-hosted OS/Python cells: pending external execution;
- Python 3.12 minimum-dependency full regression: pending;
- public highspy / OSQP / PySCIPOpt matrix: pending;
- pip-audit vulnerability/CycloneDX gate: pending;
- twine check --strict: pending;
- actual GitHub artifact attestation issuance/verification: pending;
- repository branch protection / allowed-actions / protected publish environment: unverified external state;
- license/SPDX, final name, owner metadata and project URLs: unresolved owner decisions.

The M33 milestone is therefore complete as an engineering closure, but **does not authorize publication**.
