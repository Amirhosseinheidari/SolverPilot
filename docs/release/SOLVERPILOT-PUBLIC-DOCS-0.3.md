# SolverPilot 0.3

Python 3.12–3.14. Install `pip install solverpilot` for the canonical LP/QP core,
`pip install "solverpilot[osqp,clarabel]"` for native QP/conic solving, or
`pip install "solverpilot[nlp]"` for the local nonlinear backend.

## What changed

The 0.2 numerical audit exposed scale-sensitive certificate false positives,
stale optional OSQP settings, budget-induced loss of reuse, cross-model conic
result mapping and mutable NLP results. Version 0.3 includes regression tests
and fixes for each. Existing top-level signatures remain compatible.

### Trust and accuracy

`verify_infeasibility` now accounts for the residual over the variable domain.
`verify_unboundedness` checks recession signs and QP nullspace conditions using
exact operations on the represented binary64 coefficients; an arbitrarily small
positive slope cannot satisfy a finite upper bound along an infinite ray.
`verify_optimality` corrects its lower bound for stationarity error. A conservative
strong-convexity lower bound can handle free QP variables; otherwise an unbounded
residual contribution remains unverified. These are checks of the supplied
floating-point model. Feasibility and optimality acceptance remain qualified by
the declared tolerances, not by an exact proof about unrounded real-world data.

```python
from solverpilot.applications import production_model
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.runtime import solve_verified
from solverpilot.analysis import quality_report

model = production_model([3., 2.], [[1., 1.]], [4.])
compiled = model.compile()
summary, result = solve_verified(compiled, backend=ScipyHighsLPBackend())
report = quality_report(compiled.execution_ir, result)
```

`solve_verified` raises `UnverifiedSolutionError` if an independently checked
numerical optimum is unavailable; the exception retains `summary` and `result`.
`valid_optimal` remains the legacy status for a solver-reported optimum with a
validated primal. Inspect the evidence level rather than treating that label
as independent proof. `scaling_report(problem)` reports coefficient ranges and
large-domain risks; it does not silently change units or solver tolerances.

## Sparse models and repeated solves

```python
import numpy as np
from scipy import sparse
from solverpilot.model import Model
from solverpilot.session import ReoptimizationSession
from solverpilot import SolveBudget

m = Model()
x = m.variable(1000, lower=0., upper=2.)
rhs = m.parameter(1000, value=np.ones(1000), name="capacity")
m.add(m.constant(sparse.eye(1000, format="csr")) @ x <= rhs)
m.minimize(((x - 1.) ** 2).sum())
with ReoptimizationSession(m) as session:
    first = session.solve(budget=SolveBudget(wall_time_s=10.))
    second = session.solve(updates={"capacity": np.full(1000, .5)},
                           budget=SolveBudget(wall_time_s=10.))
```

Wrap SciPy sparse matrices with `model.constant(A)` before symbolic operations.
Bare `A @ x` is not a portable symbolic dispatch path in SciPy. Affine coefficient
blocks stay CSR, and unchanged subexpressions are cached across updates. Numeric
zero/nonzero pattern changes still trigger the backend's structural checks.
Clearing an OSQP optional setting with `None` restores its native default.
Call-local budgets preserve the session-owned workspace and restore public settings.

## Convex modeling atoms

```python
from solverpilot.model import Model, norm, huber, log_sum_exp, quad_form, abs

m = Model()
x = m.variable(2, lower=-10., upper=10., name="weights")
m.add(norm(x, 2) <= 1.)
m.minimize(huber(x - m.constant([3., 4.])).sum())
compiled = m.compile()
result = compiled.solve()
original_x = compiled.reconstruct_primal(result.x)
```

Arguments must be affine. `abs` is elementwise; norm orders are 1, 2 and infinity.
Huber uses `x²` inside `[-delta, delta]` and `2*delta*abs(x)-delta²` outside.
`log_sum_exp` and `quad_form(x, P)` are scalar; `P` must be symmetric PSD.
The compiler checks convex objective/constraint direction and rejects unsupported
compositions, nonlinear equalities and integer/conic mixtures. It does not silently
turn a nonconvex expression into a relaxation. Atom graphs can introduce auxiliary
variables: use reconstruction or `named_values` for original variables. Native
results and their objectives describe the execution model, including epigraphs;
at an interrupted solve auxiliary costs need not be tight. `validate_original`
checks original constraints after reconstructing the primal.

## Common options and results

```python
from solverpilot.runtime import SolveOptions, solve_any
from solverpilot import SolveBudget, ValidationTolerances

options = SolveOptions(budget=SolveBudget(wall_time_s=10.),
                       tolerances=ValidationTolerances(feasibility=1e-8))
summary, specialized_result = solve_any(compiled, options=options)
```

The common summary exposes status, feasibility, objective, evidence level, backend
and problem identity while preserving the specialized result. LP/QP, conic, NLP,
MINLP and CP use this entry point. Unsupported guarantees raise explicit errors.
Clarabel supports common time/thread budgets, validation tolerances, iteration
progress and cooperative cancellation. NLP supports native wall time and validation
tolerances; MINLP supports its existing OA iteration limit and mapped validation
tolerance, not an aggregate hard deadline. CP-SAT maps time and worker budgets.
LP/QP hard cancellation uses process batches. No in-process API promises a hard
process deadline or enforced memory cap.

`solverpilot-capabilities --backend clarabel-native --verify` uses the same
discovery catalog as specialized solving and runs analytic runtime checks.

## Conic reuse and progress

```python
from solverpilot.conic import ClarabelBackend
from solverpilot.runtime.batch import CancellationToken

events = []
token = CancellationToken()
backend = ClarabelBackend(reuse=True)
result = compiled.solve(backend=backend, progress=events.append, cancellation=token)
```

A callback receives immutable iteration information and can return `True` to stop.
Callback exceptions stop the solve and propagate; callbacks are detached afterward.
Any retained candidate is independently validated. `reuse=True` explicitly disables
Clarabel presolve/chordal decomposition, as required by its data-update interface;
measure this tradeoff for your models. Shape, sparsity, cones or setting changes
rebuild the native solver. Raw dual/slack data are immutable execution-transport
snapshots, not independently verified conic optimality certificates.

See the [Clarabel update contract](https://clarabel.org/stable/user_guide_data_updating/),
[callback API](https://clarabel.org/stable/examples/py/example_callback/) and
[OSQP update interface](https://osqp.org/docs/interfaces/python.html).

## Streaming batches

```python
from solverpilot.runtime.batch import BatchExecutor, iter_solve_batch, solve_batch

# Use the standard `if __name__ == "__main__":` guard in scripts.
with BatchExecutor(backend="scipy-highs-ds", max_workers=4,
                   max_pending=8, timeout_s=30.) as executor:
    for item in executor.iter(problem_generator):
        consume(item)
    # The next executor.iter(...) call reuses the same processes.
```

`max_pending` bounds submission and input lookahead; yielded results need not be
retained. A timeout includes first-use process startup. A timed-out/cancelled worker
is killed and replaced before reuse. Closing a partially consumed iterator cancels
its pending jobs; always close iterators/executors when ending early. Process memory
caps remain POSIX address-space limits. Native thread counts are separate from
process concurrency. `mode="sequential"` avoids startup costs for cheap jobs but
rejects hard time/memory limits. `solve_batch` collects a tuple; `mode="isolated"`
retains the original one-process-per-job transport. A pre-cancelled sequence keeps
the legacy cancelled-item output; a cancelled stream consumes no further inputs.

## Sensitivity, scenarios and robustness

`shadow_prices(model, compiled, result)` labels canonical bound marginals with
original names. Row orientation matters: an internally represented `-x >= -b`
has the opposite bound derivative to the user parameter `b`.
`parameter_sensitivity(..., "capacity")` applies that chain rule through central
differences of compiled data and the validated dual envelope. It reports an
estimated marginal and does not guarantee differentiability or dual uniqueness.

`differentiate_qp(problem, result, dq=..., dP=..., dA=..., dl=..., du=...)`
solves a regular local KKT derivative system. Bound directions order linear rows
before variable bounds. It rejects weakly active inequalities, dependent active
rows, ill-conditioning and systems above the configured dense dimension limit
(512 by default). This is a local derivative with unchanged active set and strict
complementarity, not a derivative at every LP/QP point or a general neural-network layer.

```python
from solverpilot.scenarios import scenario_sweep, scenario_statistics

rows = list(scenario_sweep(model, [
    ("base", {}), ("more_capacity", {"capacity": [5.]}),
]))
comparison = scenario_statistics(rows)
```

Each scenario resets unspecified parameters to the initial baseline, runs in an
owned session and leaves the caller's model unchanged. Inputs are lazy; errors can
be raised or recorded. The parameter name must be unique and nonempty.

`model.robust.robust_leq` adds the exact independent-box counterpart
`a.x + radius.abs(x) <= b`. `minimize_worst_case` sets a finite-scenario minimax
linear objective. These are explicit uncertainty models, not statistical claims
about unknown distributions.

## Application templates

`solverpilot.applications` includes `production_model`, `transportation_model`,
`energy_dispatch_model` and `mpc_model`. Each returns an editable named, parameterized
Model. Production/transportation are continuous; dispatch has no network/ramping/unit
commitment; MPC uses linear dynamics, fixed PSD quadratic weights and a finite
horizon. Existing TSP/routing packs remain available. Examples are starting models,
not complete domain planning systems.

## Reproduction and development checks

```python
from solverpilot.runtime.manifest import save_run, replay_run

save_run("run.json", compiled.execution_ir, result, include_model=True)
reproduced = replay_run("run.json")
```

Model data export is opt-in. Canonical LP/QP replay checks the model hash and package
versions, restores adapter configuration and solver budget, and never loads executable
pickle data. Matching settings/versions do not imply bitwise-identical floating-point
results across hardware. Specialized runs can record metadata; executable model
replay is currently LP/QP only. An explicit backend configuration is recorded even
when a session temporarily applies a budget.

CI checks the historical signatures, new numerical scale/permutation cases, installed
wheels, optional native integrations, a coverage floor and the typed common options/
summary API. `py.typed` enables gradual type information; it does not claim that every
legacy module has complete static annotations.
