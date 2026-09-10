# Test Report — M4

Date: 2026-08-31

## Full suite

```text
98 passed, 3 skipped
```

Skipped suites are exactly the optional public native package integrations for highspy, OSQP, and PySCIPOpt. They are not counted as passes.

## Additional executed checks

```text
M1 mutation/hash properties:       1000 / 1000 passed
M2 OSQP translation properties:    1200 / 1200 passed
M3 native HiGHS reuse regression:  passed
M4 LP holdout benchmark run 1:     passed
M4 LP holdout benchmark run 2:     passed
```

The native wheel verifier itself has unit coverage for lock parsing, hashing, and missing-wheel failure behavior.
