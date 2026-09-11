# SolverPilot

> **Current version:** `0.1`. This release promotes the published `0.1.0rc2` code with no algorithm or API changes. Performance improvements are deferred to a future version. Every publication still requires exact-artifact qualification.

SolverPilot is a **trust-aware optimization runtime and modeling layer for Python**. It combines a small matrix-first solve API for LP/MILP/convex QP with a higher-level semantic modeling system that can compile into conic, smooth nonlinear, certified convex binary MINLP, and constraint-programming execution paths.

The project is deliberately conservative about claims: candidate solutions are independently validated where possible, backend capabilities are version/evidence aware, and local or backend-reported success is not silently promoted to a stronger proof category.

## What SolverPilot provides

SolverPilot has two complementary layers.

### Core runtime

- canonical `LinearProblem` / `QuadraticProblem` representations;
- LP, MILP, and continuous convex-QP execution;
- explicit or capability-aware backend routing;
- independent candidate validation and objective recomputation;
- infeasibility diagnostics;
- portfolio/fallback execution;
- repeated-solve `Session` support;
- auditable production-routing decisions with learned LP performance routing disabled.

### Extended modeling platform — Track P P0–P9

- semantic `Model`, `Variable`, `Parameter`, and expression DAG;
- parameter-aware compile cache and partial recompilation;
- backend Capability Protocol 2.0 with version-bound evidence;
- exact/restricted bridge and reformulation infrastructure with transformation tapes;
- `PersistentSession` with fail-closed native-patch vs rebuild decisions;
- SOC/RSOC/PSD conic representation and validation;
- smooth continuous NLP + automatic differentiation through the pinned CasADi verification path;
- proof-aware **certified convex binary MINLP** orchestration;
- integer/Boolean/interval constraint programming with an exhaustive reference backend and optional OR-Tools CP-SAT.

The extended APIs are canonically imported from submodules such as `solverpilot.model`, `solverpilot.conic`, `solverpilot.nlp`, `solverpilot.minlp`, and `solverpilot.cp`. The M30-frozen top-level `solverpilot.__all__` surface remains unchanged in this merge candidate.

### Verified trust, evaluation, extension, and application layers

The `0.1` release keeps the S2–S10 additive layers and adds correctness/trust hardening discovered during adversarial review of `0.1.0rc1`. The historical `0.0.40rc2` artifacts remain separate provenance and are not overwritten.

- `solverpilot.io`: strict local JSON/CSV ingestion, bounded reads, explicit mapping, content hashes, and source provenance;
- `solverpilot.evaluation`: direct-run oracle construction and decomposed quality/runtime/failure regret;
- `solverpilot.intelligence`: versioned pre-solve feature schema, leakage audit, and simple train-only OOD/shift diagnostics;
- `solverpilot.reporting`: claim-safe structured explanations for core `SolveResult`;
- `solverpilot.extensions`: explicit trusted-code extension staging without creating a second backend/bridge registry;
- `solverpilot.applications.tsp`: exact small-instance references, conservative heuristics, independent validation, MILP and restricted CP compilation;
- `solverpilot.applications.routing`: CVRP/VRPTW model, validation, timelines, diagnostics, exact small-instance reference, MILP, and conservative heuristics;
- `solverpilot.history`: explicit opt-in local SQLite history that excludes raw solution vectors and raw problem arrays by design.

These APIs remain submodule-scoped and do not change the frozen 78-symbol top-level API.

## Supported problem families and guarantee boundary

| Family | Current support | Guarantee boundary |
| --- | --- | --- |
| LP | ✅ solve | validated candidate + backend termination semantics |
| MILP | ✅ solve | validated candidate/integrality + backend termination semantics |
| Continuous convex QP | ✅ solve | validated candidate; bridge-specific optimality claims stay conservative |
| Semantic LP/MILP/QP modeling | ✅ | compiles to canonical core IR |
| Indicator constraints | ✅ restricted exact bridges | fixed-state simplification or finite-bound-certified Big-M; otherwise fail-closed |
| SOC | ✅ representation/validation; verified restricted solve path | linear-objective supported verification path only |
| RSOC | ✅ representation/validation; verified restricted solve path | exact map to SOC in verified path |
| PSD | ✅ representation/validation | generic PSD solving is not authorized by current conformance evidence |
| Smooth continuous NLP | ✅ optional | local-optimal candidate only after validation/KKT checks; **no global NLP proof** |
| Convex binary MINLP | ✅ restricted | `globally_proven` is solver-certified inside the certified P8 binary/convex scope; `independently_verified_global` remains false until the full proof chain is independently reconstructed |
| General integer/nonconvex MINLP | ❌ | fail-closed |
| Constraint programming | ✅ reference solver | exhaustive proof only within reference state budget |
| OR-Tools CP-SAT | ✅ optional | exact verified integration target: OR-Tools `9.15.6755` |
| TSP application | ✅ reference / MILP / restricted CP / heuristics | independent route validation; only exhaustive reference paths issue an independent optimality proof |
| CVRP / VRPTW application | ✅ reference / MILP / heuristics | independent route/timeline validation; exponential reference solver is for small instances only |

## Important current limitations

- The learned production solver selector remains disabled; OOD diagnostics are **not** calibrated model-confidence probabilities.
- The S4 feature schema currently targets canonical LP/MILP/convex-QP fingerprints; conic/NLP/MINLP/CP/TSP/VRP do not yet share that learned-selector feature contract.
- `solverpilot.reporting.explain_result` and the automatic history adapters are centered on core `SolveResult` / `LinearProblem` / `QuadraticProblem`; application-specific TSP/VRP results have their own validators rather than a unified report/history adapter today.
- Extension activation is transactional under the manager's documented ownership assumptions, but caller-owned `BackendRegistry` / `BridgeRegistry` objects must not be mutated concurrently outside the manager during activation.
- Local-file symlink/path checks are fail-closed for static paths, but they are not a sandbox against a hostile process racing filesystem entries between validation and open.
- `HistoryStore` is a local plaintext SQLite store. It is opt-in and minimizes persisted solver/problem payloads, but it is not encrypted and caller-supplied metadata can still contain sensitive information.
- TSP/VRP exact reference algorithms are exponential and intentionally state-budgeted; the routing layer does not yet claim pickup-and-delivery, split delivery, stochastic travel time, live traffic, or real-time redispatch support.
- The local salvage qualification environment did not contain the exact release build pins or OR-Tools runtime; those remain external release gates rather than locally claimed passes.

## Requirements

Base runtime:

- Python **3.12, 3.13, or 3.14**
- NumPy `>=2.2,<3`
- SciPy `>=1.15,<2`

Optional integrations are installed separately. OR-Tools 9.15 provides Python 3.12–3.14 wheels upstream; SolverPilot still requires its own cross-platform qualification before public support is promoted.

## Installation

### From PyPI

After the first public release:

```bash
python -m pip install solverpilot
```

### From the repository before PyPI publication

```bash
git clone <REPOSITORY_URL>
cd solverpilot
python -m pip install .
```

For development:

```bash
python -m pip install ".[test]"
python -m pytest
```

## Quick Start — LP

```python
import numpy as np
from solverpilot import LinearProblem, solve

problem = LinearProblem.from_data(
    A=[[1.0, 1.0]],
    c=[1.0, 2.0],
    variable_lower=[0.0, 0.0],
    variable_upper=[1.0, 1.0],
    constraint_lower=[1.0],
    constraint_upper=[np.inf],
)

result = solve(problem)

print(result.status.value)       # valid_optimal
print(result.objective)          # 1.0
print(result.x)                  # approximately [1.0, 0.0]
print(result.validation.valid)   # True
```

Full example: [`examples/01_lp_basic.py`](examples/01_lp_basic.py)

## Core examples

### Linear Programming (LP)

Use `LinearProblem` for continuous linear models. Lower/upper row arrays represent
`constraint_lower <= A @ x <= constraint_upper`.

Full example: [`examples/01_lp_basic.py`](examples/01_lp_basic.py)

### Mixed-Integer Linear Programming (MILP)

Integer structure is declared through `VariableDomain`.

Full example: [`examples/02_milp_binary.py`](examples/02_milp_binary.py)

### Continuous convex QP

`QuadraticProblem` uses:

```text
0.5 * x.T @ P @ x + q.T @ x + objective_offset
```

Full example: [`examples/03_qp_convex.py`](examples/03_qp_convex.py)

## Semantic modeling

The extended modeling layer builds a symbolic model and compiles it into the appropriate execution IR.

```python
from solverpilot.model import Model

m = Model("semantic-lp")
x = m.variable(2, lower=0.0, upper=5.0, name="x")
m.add(x[0] + x[1] >= 1.0)
m.minimize(2.0 * x[0] + x[1])

compiled = m.compile()
result = compiled.solve()

print(type(compiled.execution_ir).__name__)  # LinearProblem
print(result.status.value)
print(result.objective)
```

Parameters preserve semantic structure while allowing data changes and compiler-cache reuse:

```python
cost = m.parameter(2, value=[2.0, 1.0], name="cost")
```

See [`examples/08_semantic_model.py`](examples/08_semantic_model.py) and the [Extended Modeling API](docs/api/EXTENDED-MODELING-API.md).

## Conic modeling

The semantic layer supports SOC, RSOC, and PSD representation/validation.

```python
from solverpilot.model import Model
from solverpilot.conic import ConicProblem

m = Model("soc")
x = m.variable(2)
t = m.variable(lower=0.0)
m.soc(t, x)
m.minimize(t)

compiled = m.compile(use_cache=False)
assert isinstance(compiled.execution_ir, ConicProblem)
```

Representation support is intentionally broader than verified solver support. PSD and generic quadratic-conic solving are **not** automatically authorized by the current backend conformance evidence.

See [`examples/09_conic_model.py`](examples/09_conic_model.py).

## Smooth nonlinear programming (NLP)

Install the pinned verification dependency:

```bash
python -m pip install "solverpilot[nlp]"
```

Example:

```python
from solverpilot.model import Model

m = Model("nlp")
x = m.variable(lower=-4.0, upper=4.0)
m.minimize((x - 1.25) ** 4 + 0.2 * (x - 1.25) ** 2)
result = m.solve(x0=[0.0])

print(result.validation.valid)
print(result.local_optimal_candidate)
print(result.globally_proven)  # False for generic NLP
```

SolverPilot does **not** claim global optimality for generic NLPs. The optional CasADi/Ipopt path promotes a candidate only after independent primal/objective validation and recomputed KKT stationarity checks.

See [`examples/10_nlp_optional.py`](examples/10_nlp_optional.py).

## Certified convex binary MINLP

Install the NLP verification dependency:

```bash
python -m pip install "solverpilot[minlp]"
```

The current MINLP layer is intentionally narrow: binary discrete variables + certified convex supported nonlinear structure. It provides exact binary enumeration and an Outer Approximation orchestrator.

```python
from solverpilot.model import Model

m = Model("binary-minlp")
x = m.variable(lower=0.0, upper=3.0)
z = m.binary()
m.minimize((x - 2.0) ** 2 + 0.2 * (1 - z))

result = m.compile().solve()
print(result.algorithm)
print(result.globally_proven)
```

`globally_proven=True` is emitted only when the certified scope, original-space validation, and bound-closure conditions all pass. In `0.1` this is a **solver-certified proof under the compiler's convexity assumptions**, not an independently reconstructed end-to-end proof; inspect `result.proof_scope` and `result.independently_verified_global`. General integer, nonconvex, nonlinear-equality, and indicator+MINLP compositions remain fail-closed.

See [`examples/11_minlp_optional.py`](examples/11_minlp_optional.py).

## Constraint programming (CP)

The CP layer is separate from the LP/MILP runtime; CP semantics are not silently converted to MILP.

A dependency-free exhaustive reference backend is included for small models:

```python
from solverpilot.cp import CPModel

m = CPModel("tiny-cp")
x = m.int_var(0, 2, "x")
y = m.int_var(0, 2, "y")
m.add_all_different([x, y])
m.minimize(x + y)

result = m.solve()
print(result.objective)
print(result.validation.valid)
```

See [`examples/12_cp_reference.py`](examples/12_cp_reference.py).

For CP-SAT:

```bash
python -m pip install "solverpilot[cp]"
```

The integration is locked to **OR-Tools 9.15.6755** in this release candidate. The native CP-SAT solve runs in an **isolated subprocess** so an earlier SciPy/HiGHS solve cannot poison OR-Tools with a conflicting HiGHS shared-library ABI. SolverPilot transfers only canonical JSON CP IR/results across that boundary and independently revalidates the assignment/objective in the parent process. Supported CP families include integer/Boolean variables, interval variables, linear constraints, `AllDifferent`, `ExactlyOne`, allowed `Table`, `Element`, `Circuit`, `NoOverlap`, `Cumulative`, and integer min/max objectives.

See [`examples/13_cp_sat_optional.py`](examples/13_cp_sat_optional.py).

## Choosing a backend

For the matrix-first runtime, omit `backend` for deterministic capability-aware routing:

```python
result = solve(problem)
print(result.trace.backend)
```

Or request a backend explicitly:

```python
result = solve(problem, backend="scipy-highs-ipm")
```

The Capability Protocol 2.0 extended layer is available under `solverpilot.capabilities` and can represent version-bound support, restrictions, verification evidence, incremental-update channels, starts, callbacks, and result artifacts without upgrading legacy declarations to verified claims.

CLI:

```bash
solverpilot-backend-health
solverpilot-capabilities --verify
```

Full example: [`examples/04_choose_backend.py`](examples/04_choose_backend.py)

## Understanding `SolveResult`

The matrix-first `solve()` API returns `SolveResult`.

| Field | Meaning |
| --- | --- |
| `result.status` | SolverPilot public status (`PublicStatus`) |
| `result.objective` | accepted/recomputed candidate objective where applicable |
| `result.x` | candidate vector, or `None` without a candidate |
| `result.validation` | independent candidate validation report |
| `result.diagnostics` | optional infeasibility diagnostics |
| `result.backend_status` | raw backend termination text |
| `result.trace` | executed backend and phase timings |
| `result.plan` | planner decision for automatic routing |

Candidate-bearing success statuses are exposed only after validation. `INFEASIBLE`, `UNBOUNDED`, and `INFEASIBLE_OR_UNBOUNDED` normally reflect backend termination unless separate certificate/diagnostic evidence is explicitly recorded.

See [`examples/06_validation_diagnostics.py`](examples/06_validation_diagnostics.py).

## Infeasible problems and diagnostics

Diagnostics are opt-in:

```python
result = solve(problem, diagnose_infeasible=True)
print(result.status.value)
print(result.diagnostics)
```

See [`examples/05_infeasible.py`](examples/05_infeasible.py).

## Repeated solves and persistence

### Lightweight `Session`

`Session` tracks problem revisions and reports reuse conservatively.

See [`examples/07_reoptimization_session.py`](examples/07_reoptimization_session.py).

### Track P `PersistentSession`

`solverpilot.session.PersistentSession` binds a semantic `Model` to one backend lifecycle. Each update is routed through one of:

- `cold_build`
- `native_patch`
- `safe_rebuild`
- `full_rebuild`
- `no_mutation_reuse`

A native patch is selected only when the exact granular capability and persistent-lifecycle evidence are verified. Reuse is **not** treated as a universal performance win.

## Optional solver dependencies

| Extra | Install | Purpose |
| --- | --- | --- |
| HiGHS | `python -m pip install "solverpilot[highs]"` | public `highspy` native adapter |
| OSQP | `python -m pip install "solverpilot[osqp]"` | public OSQP Python adapter |
| SCIP | `python -m pip install "solverpilot[scip]"` | PySCIPOpt adapter |
| NLopt | `python -m pip install "solverpilot[nlopt]"` | NLopt SLSQP integration |
| CasADi | `python -m pip install "solverpilot[casadi]"` | pinned CasADi 3.7.2 verification bridge |
| Conic | `python -m pip install "solverpilot[conic]"` | CasADi-based conic verification path |
| NLP | `python -m pip install "solverpilot[nlp]"` | CasADi 3.7.2 + Ipopt path |
| MINLP | `python -m pip install "solverpilot[minlp]"` | dependencies for certified binary MINLP orchestration |
| CP | `python -m pip install "solverpilot[cp]"` | OR-Tools CP-SAT 9.15.6755 |
| Native bundle | `python -m pip install "solverpilot[native]"` | HiGHS + OSQP + PySCIPOpt + NLopt |
| Open-source bundle | `python -m pip install "solverpilot[open-source]"` | same open-source native solver set as the current bundle |
| Benchmarking | `python -m pip install "solverpilot[benchmark]"` | benchmark/research tooling |

An upstream package being installable is not by itself SolverPilot integration evidence. Release qualification remains platform/version specific.

## What is not supported or not claimed yet

- learned LP performance routing in production;
- nonconvex continuous QP in the matrix-first QP path;
- mixed-integer `QuadraticProblem` / general MIQP as a stable matrix-first path;
- exponential/power/generalized-power cone families;
- generic verified PSD solving;
- global optimality for generic NLP;
- nonsmooth generic NLP atoms such as arbitrary `abs/max/min`;
- general-integer or nonconvex/global MINLP;
- nonlinear equalities/two-sided nonlinear constraints in the current MINLP proof path;
- indicator+MINLP bridge composition;
- automatic guarantee that repeated solves use native warm starts/factorization reuse;
- automatic independent certificates for every infeasible/unbounded backend termination;
- a per-call validation-tolerance argument on high-level `solve()` / `solve_production()`.

See [Known Limitations](KNOWN-LIMITATIONS.md).

## Examples

Core/runtime:

1. [`01_lp_basic.py`](examples/01_lp_basic.py) — LP
2. [`02_milp_binary.py`](examples/02_milp_binary.py) — binary MILP
3. [`03_qp_convex.py`](examples/03_qp_convex.py) — convex QP
4. [`04_choose_backend.py`](examples/04_choose_backend.py) — backend selection
5. [`05_infeasible.py`](examples/05_infeasible.py) — diagnostics
6. [`06_validation_diagnostics.py`](examples/06_validation_diagnostics.py) — result validation
7. [`07_reoptimization_session.py`](examples/07_reoptimization_session.py) — repeated solves

Extended modeling:

8. [`08_semantic_model.py`](examples/08_semantic_model.py) — semantic model/compiler
9. [`09_conic_model.py`](examples/09_conic_model.py) — SOC representation
10. [`10_nlp_optional.py`](examples/10_nlp_optional.py) — optional smooth NLP
11. [`11_minlp_optional.py`](examples/11_minlp_optional.py) — optional certified binary MINLP
12. [`12_cp_reference.py`](examples/12_cp_reference.py) — dependency-free CP reference backend
13. [`13_cp_sat_optional.py`](examples/13_cp_sat_optional.py) — optional OR-Tools CP-SAT
14. [`14_persistent_session_optional.py`](examples/14_persistent_session_optional.py) — persistent semantic-model session

See [`examples/README.md`](examples/README.md).

## API Reference

- [Stable top-level Public API](docs/api/PUBLIC-API-v1.md)
- [Extended Modeling API](docs/api/EXTENDED-MODELING-API.md)
- [Current frozen top-level API snapshot](PUBLIC-API-V0_1_0RC2.json)
- [Current core backend contract](BACKEND-CONTRACT-V0_1_0RC2.json)
- [Track P merge provenance](TRACK-P-MERGE-PROVENANCE.json)

## Trust and provenance

The original frozen Track P P0–P9 bundle was verified before forward-porting:

- source bundle SHA-256: `6cc282b9428450ab7bbc39cd6c768de2da2fc092f8d7f79cbbf319f657df24a2`
- internal checksum rows: `816`
- checksum mismatches: `0`
- original Track P final version: `0.0.40`
- OR-Tools qualification target: `9.15.6755`

Frozen historical Track P documents remain under [`docs/history/track-p-p0-p9-frozen/`](docs/history/track-p-p0-p9-frozen/). They intentionally retain the old OptiMind naming as provenance; the current public brand and namespace are SolverPilot / `solverpilot`.

## Production routing

`solve_production()` remains conservative. The historical learned LP selector remains disabled; merging Track P does not authorize a new empirical performance-ranking claim.

## Development, security, and release status

- [Contributing](CONTRIBUTING.md)
- [Security policy](SECURITY.md)
- [Release process](RELEASING.md)
- [Known limitations](KNOWN-LIMITATIONS.md)

Normal CI targets Python 3.12–3.14. The exact-artifact cross-platform release qualification workflow remains the authority for public release support.

## License

SolverPilot is licensed under the [Apache License 2.0](LICENSE). Apache-2.0 permits commercial and closed-source use subject to its terms. It does not grant trademark rights.
