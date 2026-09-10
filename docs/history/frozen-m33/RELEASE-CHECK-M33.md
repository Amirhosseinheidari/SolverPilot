# Release Check M33

## Internal/software gates

- version `0.0.36rc3`: PASS
- public API remains frozen (78 symbols): PASS
- backend contract / learned LP routing OFF: PASS
- Python support policy `>=3.12`: PASS
- runtime floors NumPy `>=2.2`, SciPy `>=1.15`: PASS
- regression: PASS — 316 / 312 passed / 4 skipped / 0 failure / 0 error
- exact wheel smoke: PASS
- sdist round-trip smoke: PASS
- local NLopt/CasADi optional smoke: PASS
- exact-artifact build-once/test-many workflow design: PASS
- full-SHA Action pinning / Dependabot / attestation design: PASS
- disabled OIDC-only publish template: PASS

## External/public-RC gates

- 9/9 core GitHub-hosted OS/Python cells: **PENDING**
- Python 3.12 minimum-dependency full regression: **PENDING**
- public highspy / OSQP / PySCIPOpt integration matrix: **PENDING**
- `pip-audit` vulnerability gate + CycloneDX SBOM: **PENDING**
- `twine check --strict`: **PENDING**
- actual GitHub artifact attestation issuance/verification: **PENDING**
- branch protection / allowed-Actions / protected PyPI environment state: **UNVERIFIED EXTERNAL STATE**
- license/SPDX, final name, owner metadata and public URLs: **UNRESOLVED**

## Decision

M33 engineering milestone: **PASS / CLOSED**.  
Public RC gate: **FAIL-CLOSED / NOT AUTHORIZED**.  
Public 1.0 gate: **NOT AUTHORIZED**.
