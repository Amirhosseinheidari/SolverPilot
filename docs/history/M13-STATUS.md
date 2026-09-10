# M13 STATUS — Direct Solver C-API Verification

**Version:** 0.0.14  
**Date:** 2026-08-31  
**Release state:** Complete / verified artifact

## What M13 closes

- Direct, version/hash-locked **OSQP 0.6.3 C API** verification backend from CasADi 3.7.2 bundled library.
- Direct, version/hash-locked **HiGHS 1.10.0 C API** verification backend from CasADi 3.7.2 bundled library.
- Persistent OSQP workspace with documented vector and same-sparsity matrix update APIs.
- Persistent HiGHS model with numerical updates and explicit LP basis hot-start.
- Clean benchmark worker mode (`python -S`) so platform `sitecustomize` startup is not silently charged to solver startup.
- HiGHS process-global scheduler lifecycle handling for sequential verification.

These are **verification-only** backends. They are not the public `osqp` or `highspy` Python packages and are not registered in the default production solver registry.

## Final gates

| Gate | Result |
|---|---:|
| Unique tests | 221 |
| Passed | 218 |
| Skipped | 3 |
| Failed / errors | 0 |
| Mutation/hash checks | 1000/1000 |
| OSQP translation checks | 1200/1200 |
| MPS semantic cross-check | 200/200 |
| Diagnostics property checks | 200/200 |
| Native IIS stress | 50/50 |
| Direct OSQP C-API QP instances | 96 / 0 direct failures |
| Direct HiGHS C-API MILP instances | 100 / 0 direct failures |
| Final wheel isolated install | PASS |
| No-site worker from final wheel | PASS |
| Public highspy/osqp/PySCIPOpt install | NOT EXECUTED — DNS/egress blocked |
| MIPLIB full benchmark | NOT EXECUTED — DNS/egress blocked |
| QPLIB supported-subset benchmark | NOT EXECUTED |

## Important boundaries

1. `bundled-osqp-capi` uses solver core OSQP **0.6.3**, not public Python package OSQP 1.1.3.
2. `bundled-highs-capi` uses solver core HiGHS **1.10.0**, not public `highspy` 1.15.1.
3. M13 does not claim OSQP factorization reuse. Matrix-value updates may refactorize.
4. HiGHS scheduler reset is used only in sequential verification contexts; HiGHS documents scheduler reset during concurrent use as undefined behavior.
5. Production `@auto` still has no performance claim. Fresh trust smoke is negative on short LPs.
