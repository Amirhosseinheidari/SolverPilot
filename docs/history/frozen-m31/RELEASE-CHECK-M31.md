# Release Check M31

## Technical RC gates

- version is PEP-440 prerelease `0.0.36rc1`: PASS
- public API snapshot matches runtime: PASS
- backend contract matches builtin candidates: PASS
- Python 3.10 grammar audit: PASS
- CI compatibility workflow covers 3 OS x 4 Python core matrix: PASS (definition only)
- full regression: PASS — 291 collected / 288 passed / 3 skipped / 0 failed / 0 errors
- Windows-style clean-worker path hardening: PASS
- Linux/Python 3.13 execution: PASS after final wheel smoke
- non-executed cells remain pending, not pass: PASS
- owner metadata remains uninvented: PASS
- learned LP routing remains disabled: PASS

## Public RC / 1.0 gate

**FAIL-CLOSED / NOT AUTHORIZED** until the intended compatibility workflow executes successfully and owner publication metadata is supplied.
