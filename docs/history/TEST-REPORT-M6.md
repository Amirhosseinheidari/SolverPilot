# Test Report — M6

**Date:** 2026-08-31  
**Version:** 0.0.7

## Full suite

```text
130 collected
127 passed
3 skipped
0 failed
```

Known skips only:

1. exact public `highspy` integration;
2. exact public `osqp` integration;
3. exact public `pyscipopt` integration.

No failed tests are converted to skips.

## New M6 diagnostics tests

14 diagnostics tests cover static contradictions, elastic relaxation, native IIS normalization, runtime opt-in diagnosis/timing, deletion-filter irreducibility, integer-bound conflict localization, and check-budget behavior that must never falsely claim irreducibility.

## Property/regression checks

```text
M1 mutation/hash:                 1000 / 1000
M2 OSQP translation:              1200 / 1200
M5 MPS native cross-check:         200 / 200
M6 diagnostic planted conflicts:   200 / 200
M5 manifest references:              7 / 7
Native HiGHS IIS planted LPs:       50 / 50
```

## External real-instance check

Netlib AFIRO parses, validates, and solves to the published optimum with both available HiGHS LP algorithm bridges. A direct native HiGHS file-reader cross-check agrees with the canonical-IR objective to floating-point precision.
