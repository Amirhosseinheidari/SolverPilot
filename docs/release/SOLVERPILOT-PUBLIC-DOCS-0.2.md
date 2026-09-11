# SolverPilot 0.2

0.2 combines the runtime, reoptimization, modeling and direct conic improvements
into one release. Existing top-level names remain available. Optional parameters
extend existing signatures; new features have explicit submodule imports.

## Install

```sh
python -m pip install 'solverpilot[osqp,clarabel]==0.2'
```

Python 3.12–3.14. Base dependencies remain NumPy and SciPy. Clarabel 0.11.1 is an
optional direct binding. CasADi and CP-SAT retain their separate extras and gates.

## Accuracy and quality

```python
from solverpilot import solve, ValidationTolerances
from solverpilot.reporting import solution_quality

result = solve(problem, tolerances=ValidationTolerances(
    feasibility=1e-8, feasibility_rel=0.0))
report = solution_quality(problem, result)
```

`feasibility_rel=0` disables the relative feasibility allowance. Tolerances are
recorded in the trace. Setting tighter tolerances may reject an otherwise useful
solver result; it does not automatically tighten the native solver's settings.
Set native solver tolerances separately when necessary.

SciPy LP, native HiGHS continuous LP/QP, and native OSQP provide canonical duals
for an independent numerical KKT/gap check. These are tolerance-qualified checks,
not exact-arithmetic proofs. MILP optimality is not promoted from primal feasibility.
Inspect `result.optimality_evidence` and `raw_statistics['optimality_check']`.

`solverpilot.validate.verify_infeasibility` checks supplied Farkas multipliers.
`verify_unboundedness` checks a supplied feasible point and recession direction.
Missing certificates do not become verified termination claims. Existing OSQP
termination-certificate checks remain active.

`solution_quality` produces table-ready original row/bound violations and trust
metadata. Optional row/variable names must match the canonical matrix dimensions.
`solverpilot.model.diagnose_model` maps LP/MILP conflict and relaxation evidence
back to named semantic constraints; it never silently repairs the model.

## Repeated solves

```python
from solverpilot.model import Model
from solverpilot.session import ReoptimizationSession

m = Model()
x = m.variable(lower=0, upper=10, name='production')
demand = m.parameter(value=2., name='demand')
m.minimize((x-demand)**2)
with ReoptimizationSession(m) as session:
    first = session.solve()
    second = session.solve(updates={'demand': 3.})
```

This session owns a clone and requires unique parameter names. Update and solve
are serialized together. Failed updates roll back parameter values. A supplied
backend is owned by the session; external mutation during a solve is unsupported.
The standard `Session` and `PersistentSession` serialize their mutation/solve
methods too, but external mutation of a model owned by the caller still requires
caller synchronization. OSQP serializes numerical workspace operations.

Semantic caching benefits depend on mutation type; full matrix changes are not
guaranteed to be faster. Structure changes must rebuild incompatible solver state.

## Modeling conveniences

```python
from solverpilot.model import indexed_variables, soft_constraint, named_values

units = indexed_variables(m, ['north', 'south'], lower=0, upper=5)
soft = soft_constraint(m, units['north'] >= 6, weight=10., name='demand')
m.minimize(units['north'] + units['south'] + soft.penalty)
compiled = m.compile()
result = compiled.solve()
table = named_values(m, compiled, result)
```

Soft constraints are affine and have an explicit nonnegative penalty expression:
add it for minimization or subtract it for maximization. Existing objectives are
never modified implicitly. Existing vector variables and vector relations remain
supported. Named-value records retain IDs to disambiguate duplicate labels.

`solverpilot.runtime.multiobjective.solve_lexicographic` accepts a canonical
`LinearProblem`, ordered linear objective vectors, optional senses and a locking
tolerance. It supports LP/MILP. It advances only after a validated solver-reported
optimum and preserves the original problem. Locks permit the explicit tolerance
plus numerical feasibility tolerance; they are not exact symbolic equalities.

## Direct conic solving

```python
from solverpilot.conic import ClarabelBackend
from solverpilot.model import ExponentialCone, PowerCone

result = m.compile().solve(backend=ClarabelBackend())
```

Supported affine memberships: SOC, rotated SOC, PSD, exponential and power cones.
Convex quadratic objectives are supported. Exponential/power vectors have three
coordinates. `PowerCone(alpha)` requires `0 < alpha < 1`. Use `m.add_in_set(v, cone)`.
PSD transport uses scaled upper-triangle vectorization and enforces symmetry.
Original-space feasibility and reported objective consistency are checked.
The default conic path prefers Clarabel when installed, otherwise the existing
CasADi bridge. The latter retains its restricted SOC/RSOC scope.
Runtime conformance exercises analytic models before verified capability routing.
Generalized power cones and generic global nonconvex optimization are not included.

## Batch execution and cancellation

```python
from solverpilot.runtime.batch import CancellationToken, solve_batch

if __name__ == '__main__':
    token = CancellationToken()
    results = solve_batch(problems, max_workers=2, timeout_s=30,
                          backend='scipy-highs-ds', cancellation=token)
```

Batch accepts canonical LP/QP models, preserves input order, and owns a fresh
process per job. `max_workers` bounds simultaneous processes. The main guard is
required for scripts under spawn. Hard deadlines include process startup, with
bounded additional cleanup time. Cancelled/terminated workers cannot return an
unobserved incumbent. An optional `solver_budget=SolveBudget(...)` allows the
native solver to return a valid incumbent before the outer deadline, when its
adapter supports that limit. Limits are per job, not a total batch budget.
`memory_mb` is POSIX RLIMIT_AS address space, not RSS, and is rejected on Windows.
Do not confuse a process-count limit with native solver thread-count control.

## Timing and reproducibility

`backend_build_s` and `backend_update_s` separate adapter preparation when
instrumented; `backend_total_s` remains the complete backend call. Native SciPy
setup is inside its solve call and cannot be separated by this adapter.
Metadata caching never shares mutable solver instances globally. Call
`solverpilot.backends.refresh_backend_metadata()` after changing installations in
a live interpreter.

`tools/benchmark_02.py` records raw latency samples and repeated-QP process RSS.
Use dedicated environments, fixed thread settings and no coverage/profiler. The
synthetic measurements are not industrial-scale or long-term memory guarantees.
