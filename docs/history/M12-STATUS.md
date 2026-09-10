# M12 Status — Benchmark Trust & Release Hardening

**Version:** 0.0.13  
**Status:** RELEASE-CANDIDATE VERIFIED IN CURRENT ENVIRONMENT  
**Scope:** benchmark trust model, solver semantics, provenance, release integrity.

## Completed
- OSQP status handling distinguishes inaccurate infeasibility from proof-level status.
- Independent OSQP primal/dual infeasibility certificate checks added.
- PySCIPOpt objective offset semantics corrected; reoptimization lifecycle remains explicit.
- CasADi verification bridges pinned to the actually verified bridge package version `3.7.2`.
- Benchmark environment fingerprint tracks CPU/software/thread policy and rejects mixed timing environments by default.
- Benchmark run identity includes protocol/backend versions and instance SHA-256; task ordering is deterministic and interleaved.
- Exact-optimization summaries no longer count reference-free merely feasible candidates as successful optimal solves.
- External selector split validation is group-aware to detect train/test family leakage.
- MIPLIB/QPLIB built-in metadata was made conservative: no unverified solution-record count; QPLIB official convex count is not treated as supported-subset count.
- M10 MILP regression script now records missing third-party `p0033.mps` as `not_executed` instead of crashing.
- Portable benchmark trust smoke and installed-wheel CLI smoke completed.

## Final verification in this environment
- Pytest: **206 passed / 3 skipped / 0 failed / 0 errors** (209 unique tests).
- Mutation/hash: **1000/1000**.
- OSQP translation: **1200/1200**.
- MPS cross-check: **200/200**, max objective difference `7.105427357601002e-15`.
- Diagnostics property checks: **200/200**.
- Native HiGHS IIS stress: **50/50**.
- Wheel: `optimind_core_codename-0.0.13-py3-none-any.whl`.
- Installed wheel reports package and distribution version both `0.0.13`.

## Explicitly NOT completed
- Public `highspy`, `osqp`, `PySCIPOpt` execution in this runtime (packages unavailable).
- MIPLIB execution (fresh acquisition failed because runtime DNS/name resolution is blocked).
- QPLIB parsing/execution (supported-subset parser/filter still absent).
- Benchopt execution (not installed here; handoff remains optional).
- Fresh full M5 selector-stability run for M12: the 180-second attempt timed out and is excluded from M12 evidence.
- Production learned selector / ML policy.

## Claim boundary
M12 strengthens **measurement correctness and benchmark trust**. It does not establish a general best-solver selector.
