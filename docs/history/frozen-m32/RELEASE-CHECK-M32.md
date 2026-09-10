# Release Check M32

## Local hardening gates

- rc2 version coherence: PASS
- M31 public API preserved exactly: PASS
- backend contract preserved; learned routing off: PASS
- full regression: PASS (299/303 pass, 4 skips, 0 failures/errors)
- exact wheel build and content audit: PASS
- exact wheel isolated smoke: PASS
- sdist build and round-trip smoke: PASS
- dist manifest and rehash: PASS
- external Actions pinned to full SHA: PASS
- build-once/test-many workflow: PASS
- resolver evidence collection designed: PASS
- attestation step prepared: PASS
- active PyPI publishing disabled: PASS
- local `twine check`: NOT EXECUTED — tool unavailable; CI hard gate remains
- clean local `pip check`: NOT PASS — unrelated host package conflict; external clean CI required

## External/public gates

- 12/12 clean core OS/Python cells: **PENDING**
- optional Highs/OSQP/PySCIPOpt execution across intended matrix: **PENDING**
- license/SPDX + license files: **PENDING OWNER DECISION**
- final public distribution/brand: **PENDING OWNER DECISION**
- author/maintainer metadata: **PENDING OWNER DECISION**
- project URLs / Trusted Publisher identity: **PENDING OWNER DECISION**

## Decision

**M32 hardening milestone: PASS/CLOSED.**  
**Public RC gate: FAIL-CLOSED / NOT AUTHORIZED.**  
**Public 1.0 gate: NOT AUTHORIZED.**
