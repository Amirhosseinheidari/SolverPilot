# Test Report M10

- pytest tests: 169
- passed: 166
- skipped: 3
- failures: 0
- errors: 0

Additional numeric regressions executed on M10 source:
- mutation/hash: 1000/1000
- OSQP translation: 1200/1200
- MPS parser/native cross-check: 200/200, max objective diff 7.105427357601002e-15
- diagnostics: 200/200
- native IIS stress: 50/50

The attempted selector-stability rerun exceeded its command time window and is not counted as M10 evidence.
