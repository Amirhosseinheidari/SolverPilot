# M1 Status — Adaptive Runtime Core

**Status:** implemented and locally verified for the scope below  
**Package version:** `0.0.2`  
**Date:** 2026-08-31

## Completed in M1

| Component | Status | Notes |
|---|---|---|
| LP/MILP canonical IR | PASS | Sparse ranged form, domains, immutable normalized data |
| Convex QP canonical IR | PASS | Continuous convex minimization only |
| Structural/data hashing | PASS | Value changes vs structural changes separated |
| Independent validator | PASS | Bounds, rows, integrality, objective consistency |
| Capability manifests | PASS | Native/safe/risky/unsupported/unknown |
| Backend registry | PASS | Deterministic registration order |
| LP dual-simplex bridge | PASS | Real HiGHS via SciPy `highs-ds` |
| LP IPM bridge | PASS | Real HiGHS via SciPy `highs-ipm` |
| LP/MILP bridge | PASS | Real HiGHS via SciPy `milp` |
| QP development bridge | PASS with limitation | SLSQP; feasible candidate only, no optimality claim |
| Problem inspector | PASS | Cheap deterministic structural/numeric features |
| Capability planner | PASS | No learned/performance heuristic claims |
| Automatic `solve()` | PASS | Inspect → plan → solve → validate → trace |
| Budget semantics | PASS with limitation | Wall time where enforceable; threads/memory fail explicitly |
| Session revisioning | PASS | Versioned immutable problem updates |
| Mutation classification | PASS | Same-sparsity vs structural changes distinguished |
| Reuse assessment | PASS with limitation | Potential only; `applied=False` in M1 |
| Planner/fingerprint trace fields | PASS | Evidence level is recorded |
| Editable installation | PASS | Offline `--no-build-isolation --no-deps` |

## Verification executed

- pytest: **70 passed**;
- randomized mutation/hash property checks: **1000 passed / 1000**;
- synthetic benchmark:
  - 9 LP;
  - 10 MILP;
  - 20 convex QP;
- compileall: PASS;
- editable install: PASS.

## Synthetic benchmark observations

These are smoke observations, **not product performance claims**.

- LP DS vs IPM max objective absolute difference: `0.0`;
- median LP solve time in this small synthetic run:
  - `highs-ds`: ~`0.00787 s`;
  - `highs-ipm`: ~`0.01054 s`;
- median planner time: ~`9.04e-05 s`;
- median inspector time: ~`4.61e-04 s`;
- max QP coordinate error vs analytic optimum: ~`4.46e-07`.

The sample is too small and synthetic to justify a general rule such as “dual simplex is better.” No such rule was added to the planner.

## Important implementation decisions

1. Planner evidence level is explicitly `capability_only`.
2. Registration order is an explicit tie-break, not disguised performance intelligence.
3. SLSQP QP results are not labeled optimal.
4. Budget fields are not silently ignored.
5. Session reuse is not reported as applied until a native stateful backend actually performs it.

## Blocked / not completed

The following Python packages are absent from the execution environment:

- `highspy`;
- `osqp`;
- `pyscipopt`;
- `benchopt`.

Therefore the following remain incomplete:

- native HiGHS basis/MIP-start/reoptimization adapter;
- OSQP matrix-update/factorization reuse;
- SCIP native reoptimization/IIS;
- Benchopt benchmark orchestration;
- MIPLIB/QPLIB empirical selector evaluation.

No claim is made that these are completed.
