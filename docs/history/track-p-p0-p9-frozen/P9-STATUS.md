# P9 STATUS — Constraint Programming Core + CP-SAT

**Package:** `optimind-core-codename 0.0.40`  
**Track:** P  
**Classification:** VERIFIED RELEASE / TRACK-P FROZEN  
**Date:** 2026-09-07

P9 is losslessly rebased onto verified P8 0.0.39 and the exact OR-Tools CP-SAT 9.15.6755 binding has now been executed.

## Verified P9 scope

`IntVar`, `BoolVar`, `IntervalVar`, linear CP constraints, `AllDifferent`, `ExactlyOne`, allowed `Table`, `Element`, `Circuit`, `NoOverlap`, `Cumulative`, integer min/max objectives, independent CP validation, exhaustive reference oracle, and OR-Tools CP-SAT execution.

## Evidence

- independent P9 oracle campaigns: 500/500 PASS
- exact CP-SAT qualification: 120/120 PASS
- CP-SAT version: 9.15.6755
- exact wheel SHA-256: `ebd5aea00374e3aad7a78de59058aca5e871a26a3c385cd0860ef1d685d03c9a`
- native unified regression with CP-SAT binding installed: 460 collected / 459 passed / 1 skipped / 0 failed / 0 errors
- focused P9 passes with and without OR-Tools present

`track_p_frozen = true`
