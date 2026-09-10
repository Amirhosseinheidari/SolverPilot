# P5 STATUS — Persistent Session & Mutation Engine

**Package:** `optimind-core-codename 0.0.36`  
**Track:** P — Platform V2  
**Classification:** VERIFIED RELEASE  
**Date:** 2026-09-06

## Scope delivered

P5 binds a semantic `Model`/`CompiledModel` to one backend lifecycle and routes each model change through a fail-closed execution decision:

- `cold_build`
- `native_patch`
- `safe_rebuild`
- `full_rebuild`
- `no_mutation_reuse`

The decision consumes P2 mutation classification, P3 runtime-versioned capability evidence, and P4 transformation/certificate state. P5 does not invent structural incremental edits and does not claim a native patch unless the exact granular capability and persistent lifecycle are verified.

## Verified persistent adapters in this release

- bundled HiGHS C API through CasADi 3.7.2; HiGHS 1.10.0
  - persistent lifecycle
  - objective update
  - RHS update
  - variable-bound update
  - same-sparsity matrix-value update
  - explicit LP basis hot start
- bundled OSQP C API through CasADi 3.7.2; OSQP 0.6.3
  - persistent lifecycle
  - objective/vector update
  - constraint/variable-bound update
  - same-sparsity P/A value update
  - automatic workspace warm start is observed, but explicit `START_PRIMAL`/`START_DUAL` are **not** upgraded by P5

These are verification adapters, not claims about every public binding/version.

## Correctness evidence

- Full regression: **363 collected / 360 passed / 3 skipped / 0 failed / 0 errors**
- Persistent-vs-fresh stress:
  - HiGHS: **160/160**, zero mismatches
  - OSQP: **160/160**, zero mismatches
- Maximum HiGHS objective difference: `8.881784197001252e-16`
- Maximum OSQP objective difference: `8.881784197001252e-16`
- Maximum OSQP primal-vector difference: `2.220446049250313e-15`
- P4 indicator/certificate updates remain original-space validated under a persistent session.

## Performance evidence

Single-host smoke only; not a universal speed claim.

- HiGHS localized RHS, `n=200,m=80`: persistent/cold median ≈ **0.578** (~1.73× faster)
- HiGHS global RHS, `n=60,m=30`: persistent/cold median ≈ **1.106** (slower)
- OSQP objective QP, `n=150`: persistent/cold median ≈ **1.194** (slower)

Therefore P5 proves a correct lifecycle and exposes evidence; it does **not** make `reuse => faster` a product invariant.

## Deliberate boundaries

P5 does not yet provide:

- generic structural add/delete patching,
- public highspy/osqp/PySCIPOpt persistent conformance in this runtime,
- callbacks or lazy/user-cut execution,
- native indicator transport,
- automatic performance routing between patch/rebuild based on learned timing,
- cross-process persistent solver handles.

## Result

**P5 is VERIFIED.**

Next Track-P milestone: the blueprint's next capability layer after persistence should build on this lifecycle without weakening P0–P5 trust boundaries.
