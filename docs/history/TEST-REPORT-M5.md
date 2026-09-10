# Test Report — M5

**Version:** 0.0.6

## Contract/unit suite

Final count is recorded by `RELEASE-CHECK-M5.md` after installing the exact packaged source.

New M5 coverage includes:

- MPS basic LP parsing/solve;
- objective sense and MPS objective-offset sign;
- integer marker default bounds;
- general integer LI/UI override;
- `RANGES` semantics for G/L/E with positive/negative range;
- duplicate matrix-entry summation;
- multiple `N` row behavior;
- negative UP/default-lower-bound interpretation;
- `.mps.gz` input;
- hard rejection of semi-continuous and quadratic unsupported sections;
- unknown-row parse errors;
- direct native HiGHS MPS-reader cross-check;
- `.solu` parsing;
- `.test` manifest parsing;
- manifest runner accounting of solved/missing/parse errors;
- best-known and infeasible reference semantics;
- inconclusive reference categorization.

## Random/property/regression suites

```text
M1 mutation/hash:              1000 / 1000 passed
M2 OSQP translation:           1200 / 1200 passed
M5 MPS semantic cross-check:    200 / 200 passed
M5 manifest smoke:                7 / 7 reference matches
M3 reoptimization regression:  all expected reuse + objective checks passed
```

## Known skipped tests

Exactly three optional dependency integrations remain skipped in this environment:

- public highspy;
- public OSQP;
- public PySCIPOpt.

They must remain visible as skips until those exact packages execute.
