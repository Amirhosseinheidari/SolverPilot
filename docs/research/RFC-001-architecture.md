# RFC-001 — Adaptive Optimization Runtime for Python

**Status:** Proposed  
**Date:** 2026-08-30  
**Project codename:** OptiMind  
**Public-name status:** RISK — Microsoft Research already uses “OptiMind” for an optimization-focused language model/project. Rename before public launch is recommended.  
**Scope of v0.1:** LP, convex QP, MILP  
**Out of scope for v0.1:** CP/Scheduling, general NLP/MINLP, generic conic modeling, black-box optimization, learned policies, distributed solving.

---

## 1. Problem statement

Python already has mature modeling systems, solver APIs, numerical optimizers, and solver-independent interfaces. A new package whose main value proposition is only “one API for many optimizers” is not sufficiently differentiated.

The missing product layer we will target is a **capability-preserving adaptive runtime** that can:

1. inspect an optimization instance;
2. determine which solver features are semantically required;
3. filter installed backends by exact capability;
4. choose an execution strategy under a user-defined intent/budget;
5. perform safe reoptimization when prior state can be reused;
6. monitor attempts and execute fallbacks;
7. validate the returned solution independently;
8. diagnose infeasibility/numerical problems where possible;
9. emit a reproducible trace suitable for benchmarking and, later, learning.

The core claim is deliberately narrower than “we solve optimization better than all solvers.”  
The product claim is: **we make the optimization workflow choose, adapt, validate, diagnose, and reuse solver state automatically without erasing solver-specific capabilities.**

---

## 2. Why this direction is defensible

Existing tools already cover important neighboring layers:

- Google OR-Tools MathOpt provides a solver-independent mathematical modeling API with incremental solving, callbacks, warm starts, detailed termination reasons and solver-independent parameters.
- optimagic exposes a unified interface to many numerical optimizers and provides filters/diagnostics/benchmarking, but algorithm choice remains an explicit user workflow.
- AutoFolio/ASlib establish algorithm-selection methodology, but they are research-oriented selection infrastructure rather than a modern optimization runtime integrated with current Python solver state, validation, and reoptimization.
- Modern solvers expose materially different capabilities. Treating them as interchangeable loses performance or semantics.

Therefore the project must not be a lowest-common-denominator wrapper.

Sources:
- MathOpt: https://developers.google.com/optimization/math_opt
- optimagic: https://optimagic.readthedocs.io/en/latest/
- optimagic algorithm selection guide: https://optimagic.readthedocs.io/en/latest/how_to/how_to_algorithm_selection.html
- AutoFolio: https://github.com/automl/AutoFolio
- Algorithm-selection survey: https://arxiv.org/abs/1811.11597

---

## 3. Public naming risk

The codename **OptiMind** has a serious discoverability/branding collision.

Microsoft Research has released “OptiMind: Teaching LLMs to Think Like Optimization Experts”, with code under Microsoft OptiGuide and public model artifacts. A separate public Python optimization runtime using the same name would compete for search results, citations, package identity, and community recognition.

Action:
- keep `OptiMind` as an internal codename;
- choose a unique PyPI/GitHub/public name before public alpha;
- perform trademark/domain/PyPI/GitHub checks separately before committing to the new name.

Sources:
- https://github.com/microsoft/OptiGuide
- https://huggingface.co/blog/microsoft/optimind

---

## 4. Architectural principles

### P1 — Capability preserving, not lowest common denominator

Each backend capability is classified as:

- `NATIVE`
- `EMULATED_SAFE`
- `EMULATED_RISKY`
- `UNSUPPORTED`
- `UNKNOWN`

The planner may never silently lower a model to a weaker formulation.

Example: converting an indicator constraint into Big-M is not a transparent operation. It is permitted only when:
- the user allows the transformation;
- a finite, defensible M can be derived from bounds;
- the transformation is recorded in the trace;
- validation is performed on the original semantic model.

### P2 — Solver status is evidence, not truth

Every feasible/optimal result returned to the user is accompanied by independent validation against the canonical problem representation.

`solver_status == OPTIMAL` does **not** automatically imply `optimind_status == VALID_OPTIMAL`.

### P3 — Reoptimization is typed

There is no universal `warm_start=True`.

Reuse capabilities are represented separately:

- `SOLUTION_HINT`
- `MIP_START`
- `PRIMAL_START`
- `DUAL_START`
- `BASIS_START`
- `SAME_SPARSITY_DATA_UPDATE`
- `STRUCTURAL_INCREMENTAL_UPDATE`
- `SOLVER_NATIVE_REOPTIMIZATION`

The session engine decides whether state remains valid after each model change.

### P4 — Deterministic baseline before machine learning

v0.1 ships no learned solver selector.

The first selector is a deterministic, explainable policy based on:
- model class;
- required capabilities;
- installed backends;
- license constraints;
- hardware;
- model size/sparsity;
- repeated-solve state;
- user intent and budget.

Learning is not allowed into the release path until:
- benchmark traces exist;
- Single Best Solver (SBS) and Virtual Best Solver (VBS) baselines exist;
- leakage-safe train/test splits are implemented;
- feature extraction cost is included in evaluation.

### P5 — The runtime must remain useful with zero telemetry

No cloud dependency is required for correctness or selection.
Any future data collection must be opt-in and privacy-preserving.

### P6 — Feature extraction has a cost

A feature that takes 2 seconds to compute cannot be treated as free when selecting a solver for a 5-second budget.

Planner evaluation uses end-to-end wall clock:
`inspect + compile + backend build + solve + validate + diagnose`.

---

## 5. v0.1 system architecture

```text
User / Adapter
     |
     v
+------------------------+
| Canonical Problem IR   |
+------------------------+
     |
     v
+------------------------+
| Problem Inspector      |
| fingerprint + warnings |
+------------------------+
     |
     v
+------------------------+
| Capability Resolver    |
| required semantics     |
+------------------------+
     |
     v
+------------------------+
| Planner / Policy       |
| candidates + budget    |
+------------------------+
     |
     v
+------------------------+
| Runtime Executor       |
| attempt/fallback/race  |
+------------------------+
     |
     +------> Backend plugins
     |        HiGHS / SCIP / OSQP
     |        optional Gurobi
     |        later Clarabel/cuOpt
     |
     v
+------------------------+
| Validator              |
+------------------------+
     |
     +---- if failure/infeasible ----+
     |                                |
     v                                v
+------------------+          +------------------+
| Diagnostics      |          | Trace Store      |
+------------------+          +------------------+
     |
     v
+------------------------+
| SolveResult            |
+------------------------+
```

---

## 6. Proposed package layout

```text
src/<public_package_name>/
    __init__.py

    problem/
        protocol.py
        linear.py
        quadratic.py
        variables.py
        constraints.py
        hashing.py

    capabilities/
        enums.py
        requirements.py
        manifest.py
        resolver.py

    backends/
        base.py
        registry.py
        highs.py
        scip.py
        osqp.py
        gurobi.py          # optional extra
        clarabel.py        # phase 1.1
        cuopt.py           # experimental phase

    inspect/
        fingerprint.py
        numerics.py
        structure.py

    plan/
        intent.py
        budget.py
        candidate.py
        rules.py
        planner.py

    runtime/
        executor.py
        attempt.py
        cancellation.py
        fallback.py

    session/
        session.py
        mutations.py
        reuse.py

    validate/
        primal.py
        integrality.py
        objective.py
        dual.py            # later / partial v0.1

    diagnose/
        infeasibility.py
        scaling.py
        elastic.py

    trace/
        schema.py
        recorder.py
        hardware.py
        serialize.py

    learning/              # not imported by core in v0.1
        dataset.py
        baselines.py
        selector.py
```

---

## 7. Canonical IR

v0.1 should use a sparse numerical IR rather than recreating Pyomo.

### 7.1 LinearProblem

Canonical form:

```text
min/max  cᵀx + offset
s.t.     l_c <= A x <= u_c
         l_x <= x   <= u_x
         x_j ∈ {continuous, integer, binary}
```

Data:
- `A`: scipy sparse CSR/CSC
- `c`: float64 array
- variable lower/upper bounds
- constraint lower/upper bounds
- variable domains
- names/metadata optional
- objective sense
- immutable structural ID + mutable data version

This single ranged form maps efficiently to HiGHS and OSQP-style data structures and remains easy to validate.

### 7.2 QuadraticProblem

```text
min  1/2 xᵀ P x + qᵀx + offset
s.t. l <= A x <= u
```

v0.1 accepts only a convex continuous QP:
- `P` symmetric PSD, or
- convexity explicitly asserted by trusted adapter with optional verification policy.

Nonconvex QP is rejected in core v0.1 rather than silently routed to a global solver.

### 7.3 MILP

MILP is represented by `LinearProblem` plus variable integrality domains.

### 7.4 What is intentionally absent

No first-release symbolic algebra tree.
No generic nonlinear expression graph.
No CP interval primitive.
No SOC/PSD cone IR.

Those require different semantics and should be introduced only with explicit capability design.

---

## 8. Adapter strategy

v0.1 priorities:

### Tier A — native OptiMind sparse API
Required. This is the reference semantics used for validation.

### Tier B — file ingestion
MPS/LP import can initially be delegated where fidelity is understood, but canonical parsing should eventually be owned if cross-backend reproducibility requires it.

### Tier C — external modeling frameworks
Do not attempt all at once.

Recommended order:
1. Pyomo import adapter for LP/MILP;
2. CVXPY import adapter for convex QP;
3. MathOpt adapter;
4. PyOptInterface adapter.

Each adapter must have a fidelity test suite.
Unsupported semantic objects raise a structured error.

---

## 9. Core data types

```python
class SolveIntent(Enum):
    PROVE_OPTIMAL = ...
    BEST_FEASIBLE_UNDER_BUDGET = ...
    FAST_FEASIBLE = ...
    BALANCED = ...
    HIGH_ACCURACY = ...
    LOW_MEMORY = ...

@dataclass(frozen=True)
class SolveBudget:
    wall_time_s: float | None
    memory_mb: int | None
    threads: int | None

@dataclass(frozen=True)
class ProblemFingerprint:
    problem_class: str
    n_variables: int
    n_constraints: int
    nnz_a: int
    density_a: float
    binary_fraction: float
    integer_fraction: float
    equality_fraction: float
    coefficient_dynamic_range_log10: float | None
    objective_dynamic_range_log10: float | None
    rhs_dynamic_range_log10: float | None
    row_nnz_stats: dict
    col_nnz_stats: dict
    singleton_rows: int
    singleton_cols: int
    bounded_variable_fraction: float
    quadratic_nnz: int | None
    convexity_status: str | None
    structural_hash: str

@dataclass(frozen=True)
class SolvePlan:
    candidates: tuple
    strategy: str
    rationale: tuple[str, ...]
    transformations: tuple
    budget_allocation: dict

@dataclass
class SolveAttempt:
    backend: str
    backend_version: str
    parameters: dict
    reuse_mode: str | None
    termination: str
    timing: dict
    raw_statistics: dict

@dataclass
class ValidationReport:
    valid: bool
    max_bound_violation: float
    max_constraint_violation: float
    max_integrality_violation: float | None
    objective_recomputed: float | None
    objective_difference: float | None

@dataclass
class SolveResult:
    status: str
    solution: object | None
    objective: float | None
    plan: SolvePlan
    attempts: list[SolveAttempt]
    validation: ValidationReport | None
    diagnostics: object | None
    trace_id: str
```

---

## 10. Fingerprint v0.1

Only cheap, deterministic features belong in the default static fingerprint.

Required:
- n, m, nnz;
- matrix density;
- variable-domain counts/fractions;
- row sense counts;
- lower/upper/fixed/free variable fractions;
- row and column nnz min/median/mean/max/quantiles;
- singleton rows/columns;
- coefficient sign fractions;
- log10 range of finite nonzero matrix coefficients;
- objective coefficient range;
- finite RHS/bound range;
- objective sparsity;
- integrality fraction;
- binary fraction;
- structural hash;
- estimated model memory.

QP only:
- Hessian nnz/density;
- symmetry check;
- convexity status: `CONFIRMED`, `REJECTED`, `UNKNOWN`.

Do **not** run expensive graph embeddings or root relaxations by default.

Dynamic/pilot features are phase 2:
- presolve reduction;
- root relaxation time;
- first feasible time;
- initial primal/dual gap;
- node rate;
- bound improvement slope.

MIPLIB itself used feature-based instance characterization for representative benchmark selection, making these families of structural/statistical features a defensible starting point.

Source:
https://miplib.zib.de/Selection_Methodology.html

---

## 11. Planner v0.1

The planner is rule-based.

### 11.1 Hard filter

Remove any backend that:
- cannot represent required semantics natively or via explicitly allowed transformation;
- is not installed;
- violates license/deployment policy;
- exceeds known hardware requirements;
- cannot meet requested intent (e.g. “prove optimal” should not select a heuristic-only path as the only candidate).

### 11.2 Ranking inputs

No universal performance ranking is hardcoded before benchmarks.

Use only defensible rules such as:
- repeated convex QP + unchanged sparsity → boost backends with same-sparsity updates and state reuse;
- available prior LP basis → boost basis-capable LP path;
- MILP solution from previous session → boost MIP-start-capable backend;
- GPU/cuOpt candidate only when a compatible NVIDIA environment exists and the problem is supported;
- `PROVE_OPTIMAL` on MILP → cuOpt MIP beta cannot be the sole execution path because NVIDIA states optimality proving remains under active development.

### 11.3 Pilot/race strategy

Optional only when:
- at least two compatible backends exist;
- total budget is large enough to justify startup cost;
- no reusable solver state makes one path obviously preferable.

The initial policy can allocate a small capped pilot slice, compare progress, then continue the better candidate.
Exact pilot percentages must be learned from benchmark data, not guessed into the public API.

---

## 12. Reoptimization/session semantics

```python
session = Session(problem)
r0 = session.solve()

session.update(
    objective=q1,
    constraint_lower=l1,
    constraint_upper=u1,
)
r1 = session.solve()
```

The session records mutations:

- `OBJECTIVE_VECTOR`
- `OBJECTIVE_QUADRATIC_VALUES`
- `VARIABLE_BOUNDS`
- `CONSTRAINT_BOUNDS`
- `MATRIX_VALUES_SAME_SPARSITY`
- `MATRIX_SPARSITY_CHANGED`
- `VARIABLES_ADDED_REMOVED`
- `CONSTRAINTS_ADDED_REMOVED`
- `INTEGRALITY_CHANGED`

The reuse engine maps mutation class × backend capability → action.

Examples supported by evidence:
- HiGHS: LP basis hot start; MIP can accept partial feasible integer assignments as an initial primal bound.
- OSQP: repeated convex QPs can reuse warm-start state/factorization and update problem data.
- Clarabel: data updates are allowed when overall dimensions/sparsity remain unchanged, with documented presolve/chordal caveats.
- Gurobi: LP basis/primal/dual starts and MIP starts are distinct mechanisms.
- cuOpt: PDLP warm start is LP-only and has presolve restrictions.
- SCIP: reoptimization exists, but must be treated as a solver-specific capability rather than a generic promise.

Sources:
- HiGHS hot starts: https://ergo-code.github.io/HiGHS/dev/guide/further/
- OSQP: https://osqp.org/docs/
- Clarabel updates: https://clarabel.org/stable/user_guide_data_updating/
- Gurobi starts: https://docs.gurobi.com/projects/optimizer/en/current/reference/attributes/variable.html
- cuOpt warm start: https://docs.nvidia.com/cuopt/user-guide/latest/cuopt-python/convex/convex-examples.html
- PySCIPOpt API: https://pyscipopt.readthedocs.io/en/latest/api/model.html

---

## 13. Validation

Validation uses canonical problem data, not backend model objects.

### LP/QP
Compute:
- lower-bound violations;
- upper-bound violations;
- row lower/upper violations;
- objective from canonical data;
- compare reported vs recomputed objective.

### MILP
Additionally:
- `abs(x_j - round(x_j))` for integer variables;
- binary-domain check.

### Status policy

Example:

```text
backend: OPTIMAL
validation: FAILED
=> public status: INVALID_SOLUTION
```

For early termination:

```text
backend: TIME_LIMIT_WITH_FEASIBLE_INCUMBENT
validation: PASSED
=> public status: FEASIBLE_LIMIT
```

Tolerances must be explicit and recorded.
Default validation tolerances may initially track conservative canonical thresholds, but the benchmark suite must test them against solver-specific feasibility tolerances.

---

## 14. Diagnostics

Order of operations:

1. static numerical warnings;
2. inspect solver certificate/status;
3. validate candidate solution if one exists;
4. backend-native infeasibility analysis where available;
5. generic elastic-relaxation fallback if requested;
6. expensive deletion filtering only as an opt-in diagnostic.

Important capability differences:
- Gurobi computes IIS for continuous and MIP models.
- SCIP/PySCIPOpt exposes IIS machinery.
- HiGHS IIS is explicitly under development and currently LP-focused in its user guide.
- OSQP returns primal/dual infeasibility certificates, which are not the same thing as an IIS.
- Clarabel detects infeasibility through homogeneous embedding, not a general IIS facility.

Never label a certificate as an IIS unless it is one.

Sources:
- HiGHS: https://ergo-code.github.io/HiGHS/stable/guide/advanced/
- Gurobi: https://docs.gurobi.com/projects/optimizer/en/current/reference/python/model.html
- OSQP: https://osqp.org/docs/solver/
- Clarabel: https://clarabel.org/stable/

---

## 15. Trace schema

Every attempt must be reproducible enough to audit.

Mandatory fields:

```text
trace_schema_version
problem_structural_hash
problem_data_hash
problem_version
source_adapter
problem_class
fingerprint
transformations

backend
backend_version
backend_build_metadata_if_available
parameters
random_seed
threads

cpu_model
physical/logical_cores
ram
os
python_version
gpu_model
gpu_driver
cuda_version

time_inspect
time_compile
time_backend_build
time_solve
time_validate
time_diagnose
time_total

termination
reported_objective
recomputed_objective
primal_bound
dual_bound
gap
first_feasible_time
nodes
iterations

reuse_requested
reuse_mechanism
reuse_state_source
reuse_applied_if_observable

validation_report
warnings
error_chain
```

For iterative progress:
- timestamp;
- incumbent;
- best bound;
- relative/absolute gap;
- nodes/iterations.

---

## 16. Backend inclusion policy

### Core first-party adapters for v0.1
1. **HiGHS** — LP, convex QP, MILP; permissive and broadly useful.
2. **OSQP** — convex QP; particularly relevant to repeated parametric QPs.
3. **SCIP/PySCIPOpt** — MILP and richer future semantics; strong plugin/diagnostic surface.

### Optional v0.1 adapter
4. **Gurobi** — high-value commercial/reference backend. Must never be a required dependency.

### Phase 1.1
5. **Clarabel** — conic direction and convex QP/data-update experiments.

### Experimental later
6. **NVIDIA cuOpt** — GPU-aware path. Treat current MIP support as beta and LP/QP path separately.

Dependency strategy:

```text
package-core
package[highs]
package[osqp]
package[scip]
package[gurobi]
package[clarabel]
package[cuopt]
package[open-source]  # tested compatible set
```

The exact extras names depend on the final public package name.

---

## 17. Release gates

These are **proposed engineering gates**, not achieved results.

### Gate C0 — semantic correctness
Required before any performance claim:
- micro test corpus covers feasible, infeasible, unbounded where applicable;
- all returned feasible solutions pass canonical validation;
- no silent semantic transformation;
- adapter fidelity failures are hard errors.

### Gate C1 — cross-backend reference consistency
For small problems with known solutions:
- objective and feasibility agree within declared tolerance;
- termination-status normalization has tests;
- transformed problems validate against original semantics.

### Gate P0 — runtime overhead
Measure separately:
- inspection;
- compilation;
- backend construction;
- validation.

Initial release goal:
`runtime overhead <= max(50 ms, 5% of backend solve time)` on the agreed medium/large benchmark subset.

This is a target to test, not a current fact.

### Gate R0 — reoptimization
On workloads where a backend documents reuse:
- warm/reuse path must never silently use stale state;
- output must match cold solve within tolerance;
- total sequence wall time must be measured against cold rebuild baseline;
- ship “automatic reoptimization” claims only where median improvement is positive and confidence intervals do not show a meaningful systematic regression.

### Gate S0 — selector
Before learned selection:
- report Single Best Solver (SBS);
- report Virtual Best Solver (VBS);
- report OptiMind policy;
- feature/inspection time included;
- require positive held-out SBS→VBS gap closure before marketing the selector as performance-improving.

### Gate D0 — diagnostics
On curated conflict tests:
- every returned infeasible subsystem must itself be infeasible;
- certificates must be typed correctly;
- no feasible test case may be declared infeasible by generic diagnostics.

---

## 18. Machine learning entrance criteria

A learned selector/configurator is blocked until all are true:

- at least 3 materially different backend strategies in one problem class;
- enough traces for family-level holdout;
- SBS/VBS gap is nontrivial;
- deterministic baseline established;
- feature-cost accounting implemented;
- leakage tests implemented;
- model can abstain/fallback;
- OOD evaluation exists.

Evaluation must use group/family-aware splits rather than random near-duplicate instance splits.

BenLOC explicitly identifies standardized splits, leakage prevention and baseline choice as major issues in learning to configure MIP solvers.

Source:
https://arxiv.org/abs/2506.02752

---

## 19. Non-goals / traps

Do not:
- write simplex, barrier, branch-and-cut, or CP-SAT from scratch in v0.x;
- clone Pyomo/CVXPY syntax;
- promise one solver-independent option vocabulary for every low-level parameter;
- silently convert solver-native structures into Big-M formulations;
- add “AI” before a benchmark dataset exists;
- benchmark only easy toy instances;
- tune on the test set;
- compare multithreaded and single-threaded runs as if they were equivalent;
- count model-building/feature extraction time for competitors but not for us;
- publish a “fastest solver” claim from one hardware configuration.

---

## 20. Milestones

### M0 — Contracts
- IR
- capability enum/manifest
- normalized status model
- validator
- trace schema
- micro correctness corpus

### M1 — First backends
- HiGHS
- OSQP
- SCIP
- common time/thread/seed controls
- structured errors

### M2 — Session/reoptimization
- mutation tracking
- HiGHS basis/MIP start
- OSQP updates/warm state
- SCIP conservative reuse
- cold-vs-reuse benchmark

### M3 — Planner v0
- hard capability filtering
- intents/budgets
- explainable rule ranking
- fallback
- optional pilot strategy behind experimental flag

### M4 — Diagnostics
- scale warnings
- certificate normalization
- Gurobi optional IIS
- SCIP IIS
- HiGHS experimental IIS marked experimental
- elastic feasibility relaxation

### M5 — Reproducible benchmark publication
- Benchopt harness
- MIPLIB/QPLIB/LP suites
- raw result artifacts
- environment manifest
- performance profiles and selector baselines

### M6 — Learned policy research
Only after M0–M5 data passes gates.

---

## 21. Definition of v0.1 “done”

v0.1 is done only when:

- the three core backend adapters are tested;
- LP/convex-QP/MILP canonical validation is reliable;
- capability resolution prevents unsupported semantics;
- `Session` safely tracks reusable state;
- solve traces are reproducible;
- benchmark harness runs from a clean environment;
- public benchmark results are generated;
- README claims match measured results.

Writing code alone is not completion.

---

## 22. Evidence status

Completed in this RFC:
- architecture;
- scope;
- capability model;
- initial backend choice;
- validation/diagnostic contracts;
- benchmark release gates;
- learning entrance criteria;
- public-name risk identification.

Not completed yet:
- executable library implementation;
- empirical runtime benchmark;
- trained selector;
- empirically optimized planner rules;
- public-name legal/trademark clearance;
- customer interviews / commercial willingness-to-pay validation.
