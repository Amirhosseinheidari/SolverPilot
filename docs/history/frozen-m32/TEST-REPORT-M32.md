# Test Report M32

## Full regression

- collected: **303**
- passed: **299**
- skipped: **4**
- failures: **0**
- errors: **0**

Skips:
1. public `highspy` integration unavailable in the runtime;
2. public OSQP Python integration unavailable;
3. public PySCIPOpt integration unavailable;
4. one M27 evaluator test requires an external prepared M25 cache that is not shipped in the release source.

## Release-contract tests

M30/M31/M32 targeted API/backend/release-contract suite: **36/36 passed** after correcting one stale historical M31 version assertion. No M31→M32 API/signature drift was found.

## Exact-wheel smoke

Linux / CPython 3.13.5, installed outside source:
- LP validated optimal: pass
- MILP validated optimal: pass
- QP validation: pass
- conservative production planner: pass
- NLopt 2.11.0 backend health: pass
- CasADi 3.7.2 bridge health: pass

The exact-wheel resolver report records NumPy 2.3.5 and SciPy 1.17.0. Host `pip check` is not clean because of a pre-existing moviepy/Pillow conflict unrelated to OptiMind; the external core cell therefore remains pending.

## sdist round-trip

The M32 sdist was unpacked, rebuilt into a wheel offline, installed outside source, and passed the same core solve smoke. The rebuilt wheel is not byte-identical to the original build, but all wheel member hashes except `RECORD` are identical. No reproducible-build claim is made.
