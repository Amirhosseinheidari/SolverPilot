# M30 Status — Release Consolidation & Public API Freeze

**Package version:** `0.0.35`  
**Classification:** **FINAL VERIFIED M30 RELEASE-CONSOLIDATION — NOT PUBLIC 1.0**

M30 converts the verifiable OptiMind source line into a release-candidate-oriented product tree. It does not reopen the learned LP-selector research branch and it does not invent missing historical implementation source.

## Closed in M30

- Frozen top-level public API (`optimind.__all__`) with 78 symbols and a machine-readable signature manifest.
- Milestone/research-only evidence translators and rejected selector APIs moved to `optimind.experimental`; compatibility aliases emit `DeprecationWarning`.
- Stable public exception hierarchy for backend availability, capability mismatch, unknown backends, unsupported budgets, and no-compatible-backend planning failures.
- Machine-readable built-in backend contract and conservative production baseline policy.
- Product-facing README/changelog/release policy and explicit provenance reconciliation.
- Source-tree cleanup: historical reports moved under `docs/history`; generated build/egg-info/cache files excluded from the final source archive.
- Final wheel and sdist built offline without dependency resolution.
- Wheel installation smoke executed outside the source tree for LP, MILP, QP, production planning, error contracts, experimental namespace, backend health, and CLI help.
- sdist round-trip rebuilt a wheel and solved a validated LP outside the source tree.
- Final `MANIFEST-M30.json` and `SHA256SUMS-M30.txt` generated from the frozen release tree.
- Final M30 ZIP verified by re-reading every archived member and independently recomputing its SHA-256.

## Regression and numerical verification

Final pytest inventory:

- 283 collected
- 280 passed
- 3 skipped
- 0 failed
- 0 errors

The 3 skips are optional native-public integration tests unavailable in the active runtime; they are not counted as passes.

Additional campaigns rerun from the M30 source line:

- mutation/hash: 1000 / 1000
- OSQP same-sparsity/translation: 1200 / 1200
- infeasibility diagnostics: 200 / 200
- direct bundled HiGHS C-API MILP cross-check: 100 instances, 300 solver calls, 0 failures
- direct bundled OSQP C-API QP cross-check: 96 instances, 192 solver calls, 0 failures
- direct C-API reoptimization correctness: all objective comparisons agree and all expected reuse events were observed in the controlled workload

M30 does not create a new performance-superiority claim from the reoptimization timings.

## Production policy after M30

Learned LP performance routing remains **OFF**. Production LP routing remains conservative (`scipy-highs-ds`) unless the caller explicitly chooses another eligible backend. Historical M25–M29 research showed a DS/IPM opportunity but did not validate a transferable selector for 1.0.

## Explicit source-provenance boundary

The M30 tree contains the verifiable M13 runtime lineage plus M23–M29 planner/evidence hardening. Historical records describe M14–M22 prototypes, but the corresponding implementation bytes were not present in the later reconstructed source line. M22 official benchmark evidence is preserved; missing prototype source is not silently recreated or claimed as shipped M30 functionality.

## Remaining blockers before public 1.0

M31 must resolve/verify:

1. owner-selected license/SPDX expression and license file(s),
2. final distribution/project name,
3. author/maintainer metadata,
4. public source/docs/issues URLs,
5. clean Python/OS compatibility matrix for every version/platform advertised by 1.0.

These are **public-release blockers**, not unfinished M30 software gates. M30 itself is frozen and final-verified as the release-consolidation milestone immediately before M31.
