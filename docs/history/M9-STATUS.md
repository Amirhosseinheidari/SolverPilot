# M9 STATUS — Public cross-library backend + intent semantics

**Version:** 0.0.10  
**Date:** 2026-08-31  
**Codename:** OptiMind (internal only)

## Completed in M9

- Added `NLoptNativeBackend` using the actually installed public `nlopt 2.11.0` package.
- Continuous LP and convex-QP support through NLopt `LD_SLSQP`.
- NLopt successful stops are deliberately normalized as candidate solutions, not proof of optimality.
- Added `SolveIntent.PROVE_OPTIMAL` hard gating: adapters explicitly marked `no_optimality_certificate=True` cannot satisfy that intent.
- Added common `objective_sense` / `objective_offset` accessors on `QuadraticProblem`.
- Fixed a real QP portfolio bug discovered by wheel-level smoke testing: a successful QP attempt was being followed by a duplicate error attempt because the portfolio assumed `QuadraticProblem.objective_sense` existed.
- Added cross-library LP correctness benchmark: HiGHS DS vs NLopt SLSQP.
- Added cross-library convex-QP KKT benchmark: SciPy SLSQP vs NLopt SLSQP plus validated portfolio execution.
- Updated native-wheel lock with the official NLopt 2.11.0 CPython 3.13 Linux wheel metadata/hash.

## M9 source verification

- Pytest: **157 passed / 3 skipped / 0 failures / 0 errors**.
- Mutation/hash checks: **1000/1000**.
- OSQP translation checks: **1200/1200**.
- MPS semantic cross-check: **200/200**.
- Diagnostics property checks: **200/200**.
- Native HiGHS IIS stress: **50/50**.
- M8 fault-injection reliability regression: **100/100 health-aware valid; 100/100 portfolio recovery**.

## Cross-library evidence

### LP

- cases: **200**
- passed: **200**
- failed: **0**
- max objective difference vs HiGHS DS: **3.144e-13**
- max NLopt feasibility violation: **2.203e-13**

### Convex QP

- instances: **48**
- direct solver calls: **288**
- portfolio executions: **48**
- portfolio backend attempts: **96**
- SciPy failures: **0**
- NLopt failures: **0**
- validated portfolio results: **48/48**
- max objective error vs constructed KKT optimum: **2.724e-11**
- max coordinate error vs constructed KKT optimum: **2.473e-06**
- timing wins: SciPy **48**, NLopt **0**.

Timing result is not a general solver claim. Both adapters use SLSQP-family local methods; M9 uses them to validate cross-library execution and trust boundaries.

## Existing performance evidence re-run from M9 source

- selector synthetic LP policy/SBS: **0.931**; bootstrap 95% **[0.890, 0.974]**; gap closure **0.876**.
- repeated-LP stateful/cold median wall ratios:
  - small: **0.646**
  - medium: **0.430**
  - large: **0.342**

## Public backend health from installed M9 wheel

- `highspy-native`: **unavailable**
- `osqp-native`: **unavailable**
- `pyscipopt-native`: **unavailable**
- `scipy-highs-ds`: **healthy** (1.17.0)
- `scipy-highs-ipm`: **healthy** (1.17.0)
- `scipy-highs-bridge`: **healthy** (1.17.0)
- `scipy-slsqp-qp-bridge`: **healthy** (1.17.0)
- `nlopt-slsqp-native`: **healthy** (2.11.0)

## Still not completed

- public `highspy` execution in this runtime;
- public OSQP execution/update/warm-start;
- public PySCIPOpt execution/IIS;
- cross-solver SBS/VBS using specialized solver families;
- full Netlib;
- MIPLIB 2017;
- supported QPLIB subset;
- learned selector;
- public package name.

M9 is the first milestone with an actually installed external optimization library outside SciPy/HiGHS (`nlopt`) participating in the public backend registry and portfolio. It does **not** substitute for the still-missing HiGHS/OSQP/SCIP public integrations.
