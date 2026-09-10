# Test Report — M1

**Date:** 2026-08-31  
**Version:** `0.0.2`

## Environment

```text
Python 3.13.5
Linux 6.18.35 x86_64
NumPy 2.3.5
SciPy 1.17.0
pytest 9.0.2
setuptools 82.0.1
wheel 0.46.3
```

Optional solver/benchmark packages checked and absent:

```text
osqp: not installed
highspy: not installed
pyscipopt: not installed
benchopt: not installed
```

## Unit / contract / integration suite

Command:

```bash
pytest -q
```

Result:

```text
70 passed
```

Coverage categories include:

- canonical LP/MILP construction;
- convex-QP construction/convexity rejection;
- structural and data hashing;
- capability requirements and risky-emulation rejection;
- canonical solution validation;
- backend registry and trace serialization;
- SciPy/HiGHS LP/MILP execution;
- HiGHS dual-simplex and IPM LP bridges;
- development QP bridge semantics;
- inspector features and edge cases;
- planner capability filtering/tie behavior;
- automatic solve routing;
- budget enforcement/failure semantics;
- Session revisions and mutation classification;
- reuse-assessment semantics.

## Randomized property checks

Command:

```bash
PYTHONPATH=src python benchmarks/m1_property_checks.py
```

Result:

```text
250 seeds
1000 checks
1000 passed
```

For each generated LP, checks covered:

1. objective-only update preserves structural hash;
2. same-sparsity matrix-value update preserves structural hash and is typed correctly;
3. removing a nonzero changes sparsity and structural hash;
4. no-op update does not increment session revision.

## Synthetic solve smoke benchmark

Command:

```bash
PYTHONPATH=src python benchmarks/m1_smoke.py
```

The benchmark solved and validated:

- 9 random bounded sparse LPs with equality/ranged-row coverage using both HiGHS dual simplex and HiGHS IPM;
- 10 random binary knapsack MILPs through SciPy/HiGHS;
- 20 diagonal convex QPs with known analytic optimum through the development SLSQP bridge.

All benchmark assertions passed.

Detailed raw output:

`benchmarks/results/m1-smoke.json`

## Build/import verification

Commands:

```bash
python -m compileall -q src tests benchmarks
python -m pip install -e . --no-build-isolation --no-deps
```

Both completed successfully in the current environment.

## Bugs found during this development cycle

### Empty-constraint SLSQP bridge

SciPy SLSQP conversion fails if an empty `LinearConstraint` is passed for a zero-row QP. The bridge was changed to pass no constraint object when `m == 0`.

### Budget semantics

The initial M1 auto-solve implementation recorded `SolveBudget` in the plan but did not enforce it at the backend boundary. This was fixed before release packaging:

- wall time is cloned into backends that expose a real `time_limit_s`;
- unsupported memory/thread limits now raise instead of being ignored.

These fixes are covered by tests.

## Not tested

Because dependencies are not present, this report does **not** cover:

- native `highspy` APIs;
- OSQP updates or warm starts;
- PySCIPOpt/SCIP;
- Benchopt;
- MIPLIB or QPLIB datasets;
- GPU/cuOpt paths.
