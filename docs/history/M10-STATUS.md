# M10 Status — 0.0.11

## Completed
- Added verification-only CasADi bridges for OSQP, HiGHS and CBC.
- Active health probe passes for all three bridges in the current environment.
- Cross-solver convex-QP benchmark: 48 instances, 576 direct solves, zero failures across four implementations.
- Cross-solver binary-MILP benchmark: 100 exact-bruteforce instances plus public p0033; zero failures across three solver paths.
- Stateful CasADi-OSQP keeps a same-structure solver object and passes the prior primal solution as x0.
- CasADi bridges are `verification_only=True` and intentionally excluded from `default_registry()`.
- CasADi-OSQP/CBC wall-time budgets are rejected because mappings are not verified; CasADi-HiGHS has verified wall-time mapping.

## Test gate
- pytest: 169 total, 166 passed, 3 skipped, 0 failed, 0 errors.
- mutation/hash 1000/1000; OSQP translation 1200/1200; MPS 200/200; diagnostics 200/200; IIS 50/50.

## Still not completed
- dedicated public Python highspy 1.15.1 execution.
- dedicated public Python osqp 1.1.3 execution.
- dedicated public Python PySCIPOpt 6.2.1 execution.
- full MIPLIB/QPLIB/Netlib benchmark suites.
- production empirical cross-solver planner policy.

## Claim boundary
CasADi bridges prove real execution against solver engines bundled with CasADi 3.7.2. They do not prove the dedicated Python-package adapters. OSQP reuse evidence means a persistent CasADi Function plus explicit previous `x0`; no factorization-reuse claim is made.
