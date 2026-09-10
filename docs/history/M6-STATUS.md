# M6 Status — Typed Diagnostics and First Real Netlib Smoke

**Date:** 2026-08-31  
**Version:** 0.0.7  
**Status:** Engineering/research milestone completed within current package/network constraints.

## What M6 actually completed

### 1. Typed infeasibility diagnostics

M6 adds a `optimind.diagnose` layer that deliberately distinguishes evidence types:

- proof-producing static contradictions;
- native IIS evidence when a backend actually exposes an IIS API;
- phase-I elastic relaxation;
- solver-agnostic deletion-filter conflict localization.

The implementation does **not** label elastic relaxation or deletion filtering as a native IIS.

Static proofs currently include:

- empty constraint rows whose fixed activity 0 lies outside the row bounds;
- integer/binary variable intervals containing no integer value.

`elastic_relaxation(problem)` gives every finite row and variable bound side its own nonnegative slack, preserves original integrality, and minimizes weighted violation. An optimal strictly positive phase-I objective proves that the original model has no zero-slack feasible solution.

`deletion_filter_conflict(problem)` starts from every finite row/variable-bound side and removes unnecessary atoms through repeated feasibility solves. It reports `irreducible=True` only after every retained atom has been proven necessary. If its check/time budget is exhausted it explicitly reports `complete=False` / `irreducible=False`.

### 2. Optional runtime diagnosis

`solve(..., diagnose_infeasible=True)` now attaches an `InfeasibilityReport` when a linear/MILP backend returns `INFEASIBLE`. Diagnosis time is recorded separately in `trace.timings.diagnose_s`.

The default remains `False` so production solves do not pay diagnostic overhead unexpectedly.

### 3. Diagnostic property verification

A deterministic randomized verification generated 200 planted infeasible models:

- 100 row-conflict LPs with irrelevant extra constraints;
- 100 integer-bound conflicts.

Result:

```text
row conflicts:             100 / 100
integer bound conflicts:   100 / 100
total:                     200 / 200
max deletion checks:       11
```

Additionally, the development native HiGHS IIS path was stress-checked on 50 planted infeasible LPs: **50/50** returned valid IIS evidence with the expected conflicting rows represented.

### 4. First externally sourced real Netlib smoke

M6 ran the public Netlib **AFIRO** instance without redistributing the instance in the artifact. The MPS content was taken from the public COIN-OR Data-Sample mirror and the published reference optimum was taken from the Netlib summary table.

Canonical parser + runtime:

```text
variables:         32
constraints:       27
canonical A nnz:   83
DS objective:      -464.7531428571429
IPM objective:     -464.7531428571429
reference:         -464.75314286
abs error:         2.857e-09
```

The same temporary MPS file was also read directly by the native HiGHS file reader and compared against OptiMind's canonical parse/solve path:

```text
native objective:        -464.75314285714285
canonical IR objective:  -464.7531428571429
absolute difference:     5.684e-14
canonical validation:    True
```

This is the first real public-instance execution in the project. It is a **smoke**, not a representative Netlib performance study.

### 5. Prior selection and reoptimization evidence re-run

M5 selector stability was rerun after M6 changes:

```text
instances:            24
solver calls:         240
DS wins:              13
IPM wins:             11
policy/SBS ratio:     0.924571
bootstrap 95% CI:     [0.880347, 0.970346]
gap closure:          0.878841
```

The production planner remains `capability_only`; synthetic within-HiGHS selection evidence is still not promoted to a production policy.

M3 stateful HiGHS reoptimization was rerun separately from scratch after an earlier combined command hit the shell timeout. The complete rerun produced:

```text
small wall ratio:    0.603443
medium wall ratio:   0.412012
large wall ratio:    0.333424
all objectives agree: yes
all expected reuse applied: yes
```

The timed-out partial run is not used as evidence.

## Still not completed

- public `highspy 1.15.1` package execution;
- public OSQP package execution;
- public PySCIPOpt package execution;
- full Netlib corpus execution;
- MIPLIB 2017 execution;
- QPLIB execution;
- Benchopt execution;
- cross-solver production selection;
- learned selector/configurator;
- public package naming/legal clearance.

Package download attempts still fail because this runtime cannot resolve/reach PyPI from Python/pip. Those missing native integrations remain explicit skips.

## Gate decision

M6 establishes that the runtime can now do more than return an infeasible status: it can produce typed, auditable evidence and a bounded solver-agnostic conflict fallback. It also moves public-corpus testing from "format ready" to **one real Netlib instance executed correctly**, without pretending that one tiny instance validates general performance.
