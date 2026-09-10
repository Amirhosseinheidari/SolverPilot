# M31 Status — Technical Release-Candidate Compatibility Checkpoint

**Version:** `0.0.36rc1`  
**Classification:** VERIFIED LINUX/PYTHON-3.13 TECHNICAL RC CHECKPOINT — NOT PUBLIC RC / NOT 1.0

M31 preserves the M30 frozen public API and backend contract, adds a fail-closed cross-platform compatibility protocol, and prepares the CI matrix needed for public compatibility claims.

## Verified in the active runtime

- Linux x86-64 / CPython 3.13
- full regression: 291 collected / 288 passed / 3 skipped / 0 failed / 0 errors
- NLopt 2.11 and CasADi 3.7.2 optional paths healthy on Linux/Python 3.13
- Windows-style clean-worker path filtering regression fixed and tested
- Python-3.10 grammar compatibility across all package source files
- M30 public API remains frozen
- conservative production planner remains enabled; learned LP performance routing remains disabled
- wheel/sdist build and isolated current-runtime smoke (recorded by M31 verification)

## Not verified in this runtime

- CPython 3.10 / 3.11 / 3.12 runtime behavior
- Windows runtime behavior
- macOS runtime behavior
- all optional native solver combinations across the matrix

The environment could not download additional CPython runtimes because DNS resolution to GitHub failed. Those cells remain `pending_ci`, not pass.

## Public 1.0 blockers

- compatibility matrix execution
- license/SPDX + license text
- final project/distribution naming decision
- author/maintainer metadata decision
- public project URLs

M31 must not be interpreted as a public 1.0 release authorization.
