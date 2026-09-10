# Current Milestone Status

Date: 2026-08-30

## Core-contract M0 — PASS

Implemented and executed:
- canonical sparse LP/MILP IR;
- continuous convex-minimization QP IR;
- contract checks for dimensions, NaN, finite matrix coefficients, bounds and domains;
- read-only numerical inputs;
- structural/data hashing;
- QP convexity verification;
- capability/support-level model;
- capability requirement extraction/filtering;
- backend protocol/registry;
- independent solution validator;
- normalized public-status vocabulary;
- versioned trace foundation;
- runtime executor with validation gate.

## Executable bridge — PASS

Because native `highspy` is unavailable in the current container and outbound binary download is blocked, a temporary `ScipyHighsBackend` was implemented using `scipy.optimize.milp` / its bundled HiGHS solver.

Actually tested end to end:
- bounded LP minimization;
- LP maximization with objective offset;
- binary MILP;
- infeasible LP;
- unbounded LP;
- capability rejection of unsupported QP.

Independent brute-force cross-check:
- 100 random binary MILPs;
- every feasible case matched the exhaustive optimum;
- every infeasible case matched exhaustive infeasibility;
- 100/100 passed.

## Native HiGHS M1 — BLOCKED BY ENVIRONMENT, NOT CLAIMED

Not implemented/tested yet:
- native `highspy` model construction;
- LP basis extraction/injection;
- MIP starts;
- native callbacks/progress;
- incremental model editing;
- native IIS path.

PyPI currently publishes `highspy 1.15.1` for CPython 3.13/Linux, but the execution environment blocks the binary download path. The repository therefore does **not** claim native HiGHS support yet.

## Other M1+ work not done

- OSQP adapter;
- SCIP adapter;
- Session/reoptimization;
- inspector feature vector;
- planner;
- diagnostics;
- Benchopt harness;
- MIPLIB/QPLIB benchmarks;
- learned selector.

No performance claim is justified yet.
