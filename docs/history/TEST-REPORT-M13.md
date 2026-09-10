# TEST REPORT M13

## Final pytest inventory

- **221 unique tests**
- **218 passed**
- **3 skipped**
- **0 failed / 0 errors**
- Duplicate test IDs across batches: **0**

Batches:

```json
{
  "cli": {
    "tests": 3,
    "failures": 0,
    "errors": 0,
    "skipped": 0
  },
  "core": {
    "tests": 196,
    "failures": 0,
    "errors": 0,
    "skipped": 3
  },
  "harness": {
    "tests": 11,
    "failures": 0,
    "errors": 0,
    "skipped": 0
  },
  "m13": {
    "tests": 11,
    "failures": 0,
    "errors": 0,
    "skipped": 0
  }
}
```

The three skips are the dedicated public-package integration modules for `highspy`, `osqp`, and `PySCIPOpt`, which are absent from this runtime.

## Fresh numerical regressions

- Mutation/hash: **1000/1000**
- OSQP translation: **1200/1200**
- MPS parser vs native HiGHS reader: **200/200**, max objective diff `7.105427357601002e-15`
- Diagnostics property checks: **200/200**
- Native HiGHS IIS stress: **50/50**

## M13-specific regression found and fixed

Repeated mixed health probing intermittently made the direct HiGHS C-API backend return `kHighsStatusError (-1)`. The root cause is HiGHS' **process-global scheduler** interacting with multiple HiGHS bindings/thread configurations in one Python process.

The final implementation:

- binds `Highs_resetGlobalScheduler`;
- resets the scheduler before constructing an independent verification C-API instance when a thread count is set;
- closes stateful native health-probe backends after probing;
- has a regression that repeats mixed built-in health probes three times and requires direct HiGHS to remain healthy.

The final JUnit files were regenerated **after** this fix.
