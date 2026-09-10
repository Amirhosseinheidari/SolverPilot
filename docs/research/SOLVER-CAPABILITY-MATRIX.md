# Solver Capability Matrix — Research Baseline

**Date:** 2026-08-30  
**Purpose:** Design input for the adaptive runtime.  
**Important:** This is a conservative engineering matrix, not a solver leaderboard.

Legend:
- ✅ = documented and suitable for the capability category
- ⚠️ = supported with important limitations / beta / interface caveat
- ❌ = not supported for this role
- ? = not sufficiently confirmed for us to rely on in v0.1

| Backend | LP | Convex QP | MILP | MIQP | QCQP / Conic | General NLP | Model/data update | Reuse / warm start | Runtime callback / interrupt | Infeasibility tooling | License / deployment note |
|---|---|---|---|---|---|---|---|---|---|---|---|
| HiGHS | ✅ | ✅ | ✅ | ❌ | ❌ | ❌ | ✅ model edits | ✅ LP basis; ✅ MIP partial solution | ✅ | ⚠️ IIS under development; guide says currently LP-focused | MIT |
| OSQP | ⚠️ LP can be represented as QP, but not why we include it | ✅ convex QP | ❌ | ❌ | ❌ | ❌ | ✅ efficient updates; fixed sparsity constraints apply to matrix-value updates | ✅ warm start; cached factorization for parametrized problems | ⚠️ not a rich MIP-style callback surface | ✅ primal/dual infeasibility certificates; ❌ IIS | Apache-2.0 |
| SCIP / PySCIPOpt | ✅ | ⚠️ can model quadratic/nonlinear structures, but v0.1 adapter uses it primarily for MILP | ✅ | ✅ | ✅ via SCIP nonlinear/quadratic machinery | ✅ nonlinear/MINLP capability | ✅, but solver stages matter | ✅ solution injection; ⚠️ native reoptimization is solver-specific | ✅ deep plugin/event surface | ✅ IIS machinery | Apache-2.0 since SCIP 8.0.3; linked optional deps may have other licenses |
| Gurobi | ✅ | ✅ | ✅ | ✅ | ✅ convex/nonconvex quadratic and SOC-recognizable forms | ✅ current nonlinear constraint support; global spatial B&B path documented | ✅ | ✅ LP basis/primal/dual starts; ✅ MIP start | ✅ rich callback API | ✅ IIS for continuous and MIP | Commercial; academic options exist; never core-required |
| Clarabel | ✅ conic form | ✅ | ❌ | ❌ | ✅ SOC, exponential, power, PSD cones | ❌ general nonconvex NLP | ✅ if shape/sparsity pattern unchanged; caveats with presolve/chordal | ? do not promise generic warm-start in v0.1 | ✅ termination callback | ✅ infeasibility detection; ❌ general IIS | Apache-2.0 |
| NVIDIA cuOpt 26.08 | ✅ | ✅ | ⚠️ beta | ❌ current MIP scope is linear objective/constraints | ⚠️ QCQP/SOCP beta | ❌ | ⚠️ API-specific; benchmark before claims | ⚠️ PDLP warm start for LP; not MIP; presolve caveat | ? do not depend on callback semantics in v0.1 | ⚠️ solver statuses/certificates must be mapped separately | NVIDIA GPU stack; MIP beta; deployment/licensing review required |

---

## Important design consequences

### 1. No universal warm-start bit

Backend reuse is not semantically uniform.

HiGHS:
- an LP basis drives simplex hot start;
- a partial feasible integer assignment can seed a MIP primal bound;
- large LP modifications may make a scratch solve with presolve/IPM preferable.

OSQP:
- specifically attractive for repeated convex QPs because factorization can be cached and solutions warm-started.

Clarabel:
- data updates require the problem shape/sparsity to remain fixed;
- update is restricted when certain presolve/chordal settings are active.

Gurobi:
- basis/primal/dual starts for LP and `Start` for MIP are distinct.

cuOpt:
- PDLP warm start is currently LP-only; documented examples require care around presolve.

Therefore the runtime API exposes typed reuse capabilities.

### 2. IIS and “infeasibility evidence” must be distinct

An IIS, a dual/primal infeasibility certificate, and an infeasible solver status are different objects.

Never normalize them into one fake `conflicting_constraints` field.

### 3. Gurobi is a useful reference backend, not a hard dependency

Its broad model surface and diagnostics make it useful in benchmarking and validation, but a public open-source package should remain fully useful without a commercial solver.

### 4. cuOpt should enter as a hardware-aware experimental path

NVIDIA cuOpt 26.08 documents:
- LP and QP support;
- QCQP/SOCP beta;
- MIP beta;
- MIP currently focused on finding high-quality feasible solutions quickly, with proving optimality still under active development;
- concurrent LP mode running GPU PDLP, GPU barrier, and CPU dual simplex, returning the method that finishes first.

That is strong evidence for hardware-aware/adaptive planning, but not evidence that every problem should go to GPU.

---

## Primary sources

### HiGHS
- Advanced / IIS:
  https://ergo-code.github.io/HiGHS/stable/guide/advanced/
- Hot starts:
  https://ergo-code.github.io/HiGHS/dev/guide/further/
- Callbacks:
  https://ergo-code.github.io/HiGHS/dev/callbacks/
- General API:
  https://ergo-code.github.io/HiGHS/dev/

### OSQP
- Overview/features:
  https://osqp.org/docs/
- Python:
  https://osqp.org/docs/interfaces/python.html
- Infeasibility:
  https://osqp.org/docs/solver/

### SCIP / PySCIPOpt
- PySCIPOpt model API:
  https://pyscipopt.readthedocs.io/en/latest/api/model.html
- SCIP:
  https://www.scipopt.org/
- License:
  https://www.scipopt.org/download.php

### Gurobi
- Constraints:
  https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/constraints.html
- Objectives:
  https://docs.gurobi.com/projects/optimizer/en/current/concepts/modeling/objectives.html
- Variable starts:
  https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html
- IIS / Python model:
  https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
- Callbacks:
  https://docs.gurobi.com/projects/optimizer/en/current/reference/cpp/callback.html

### Clarabel
- Supported cones:
  https://clarabel.org/stable/api_cone_types/
- Data update:
  https://clarabel.org/stable/user_guide_data_updating/
- Termination callback:
  https://clarabel.org/stable/user_guide_callbacks/
- Overview:
  https://clarabel.org/stable/

### NVIDIA cuOpt
- Introduction:
  https://docs.nvidia.com/cuopt/user-guide/latest/introduction.html
- Convex features:
  https://docs.nvidia.com/cuopt/user-guide/latest/convex-features.html
- MIP features:
  https://docs.nvidia.com/cuopt/user-guide/latest/milp-features.html
- Release notes:
  https://docs.nvidia.com/cuopt/user-guide/latest/release-notes.html
