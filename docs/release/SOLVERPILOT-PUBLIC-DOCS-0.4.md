# SolverPilot 0.4

This release implements the qualified CPU paths from the 0.3 research audit.
It repairs the near-indefinite QP certificate counterexample, adds an explicit
SCIP global path, generalized power cones, CPU PDLP, LP witness recovery and
observable reuse. The 78 top-level names remain compatible. `solve()` and
`solve_production()` gain the optional keyword `certificate_recovery`.

## Installation and entry points

```bash
python -m pip install "solverpilot[global,pdlp,clarabel,osqp]==0.4"
solverpilot-capabilities --backend scip-global --verify
solverpilot-capabilities --integrations
```

Base installation still requires only NumPy and SciPy. Optional packages must
be available for the Python/platform combination in use. New backends are
explicit opt-ins; neither PDLP nor global SCIP enters the conservative default
LP/QP selection policy. CPU PDLP runs in an isolated worker so its OR-Tools
native libraries do not collide with HiGHS loaded by another adapter.

## Scope against the research checklist

| Research item | 0.4 implementation | Boundary |
| --- | --- | --- |
| Learned LP routing | `observe_shadow_policy` measures suggestions and overhead | Production selection remains unchanged; no demonstrated held-out speed advantage |
| Nonconvex continuous QP | `GlobalQuadraticProblem` + `SCIPGlobalBackend` | Explicit separate IR; numerical global solver evidence |
| Matrix MIQP | Same IR with integer/binary domains | Original quadratic objective and integrality validated; no independent generic global proof |
| Exponential/power/generalized power | Clarabel with model, IR and original-space validation | Transcendental cone dual membership is not independently certified |
| Verified PSD solving | Conservative original-domain conic lower bounds | Exact PSD evidence required; unsupported matrices return unverified |
| Global NLP | Bounded factorable expressions through SCIP | Finite variable bounds and documented expression whitelist; no arbitrary callbacks |
| Nonsmooth NLP | Explicit `absolute`, `maximum`, `minimum` global atoms | Algebraic SCIP reformulations; smooth Ipopt path is unchanged |
| General integer/nonconvex MINLP | `FactorableProblem` / `Model.compile(target="global")` | Numerical global branch-and-bound evidence; existing convex binary OA proof scope unchanged |
| Nonlinear equality/two-sided rows | Global compiler accepts equality and `Interval` | Accepted in the global path, not silently added to the OA proof path |
| Indicator + MINLP | Binary premise with exact rational interval-derived Big-M | Finite provable enclosure required; original active body revalidated |
| Repeated solves | OSQP skips matrix updates for vector-only edits; common reuse view | Workspace reuse does not establish native factorization reuse |
| Infeasible/unbounded certificates | Optional time- and size-bounded LP witness recovery | No universal success guarantee; unsupported or failed recovery is unverified |
| Per-call validation tolerance | `ValidationTolerances` on `solve`/`solve_production` and common options | Validator tolerance is distinct from solver stopping tolerances |

## Near-indefinite QP repair

For the represented problem `P=[[-1e-12]]`, `q=[0]`, `-1e10 <= x <= 1e10`,
the point `x=0` is stationary but an endpoint has objective `-50,000,000`.
It must never receive an independent convex-QP optimality label.

`certify_psd()` now checks exact represented coefficients using rational
Gershgorin bounds or rational LDL decomposition. Negative diagonals and a zero
diagonal with a nonzero off-diagonal witness reject PSD immediately. LDL is
limited to dimension 32 and bounded intermediate arithmetic; sparse diagonal
dominance can certify larger matrices. An inconclusive certificate is `unknown`.
Numerical positive eigenvalues may establish backend eligibility but cannot
substitute for PSD evidence in an independent bound.

## Matrix nonconvex QP and MIQP

```python
import numpy as np
from solverpilot.globalopt import GlobalQuadraticProblem, solve_global

p = GlobalQuadraticProblem.from_data(
    P=[[0, -1], [-1, 0]], q=[0, 0], A=np.empty((0, 2)),
    variable_lower=[-2, -2], variable_upper=[2, 2],
    constraint_lower=[], constraint_upper=[],
    domains=["integer", "continuous"],
)
r = solve_global(p)
print(r.objective)  # -4
print(r.validation.valid)
print(r.raw_statistics["solver_dual_bound"])
```

The convention is `0.5*x.T@P@x + q.T@x + objective_offset`. Both senses are
supported. Existing `QuadraticProblem` stays continuous and convex: choosing a
global solver does not reinterpret or weaken that contract.

## Factorable global NLP/MINLP

```python
from solverpilot.model import Model, Interval
from solverpilot.globalopt import SCIPGlobalBackend, absolute, solve_global

m = Model()
x = m.variable(lower=-2, upper=2)
z = m.variable(lower=0, upper=4, domain="integer")
enabled = m.variable(lower=0, upper=1, domain="binary")
m.add_in_set(x*x, Interval(0.25, 4))
m.indicator(enabled, x*x <= 1)
m.minimize((x*x - 1)**2 + 0.1*x + (z - 2.3)**2 + 0.01*absolute(x))
r = solve_global(m, backend=SCIPGlobalBackend(time_limit_s=30, threads=1))
```

Supported nodes: constants, parameters, variables, indexing, transpose, sum,
concatenation, arithmetic, matrix products, integer powers from 0 to 64,
sin/cos/exp/log/sqrt/tanh and the three explicit nonsmooth atoms. `maximum(x)`
and `minimum(x)` reduce an array; an optional second argument is elementwise.
`absolute` accepts nonlinear expressions. The existing affine `model.abs` also
works in this global path.

All variables in factorable models need finite declared bounds. Division must
exclude zero across its denominator interval; log arguments must be positive;
sqrt arguments must be nonnegative. These domains are checked before native
solving. Expressions whose domain cannot be certified are rejected. Indicator
bodies need an enclosure computed with exact rational arithmetic and outward
rounding; some transcendental compositions therefore remain unsupported.

Compile with `m.compile(target="global")`, or select `SCIPGlobalBackend` through
`m.solve()`. Parameters are snapshotted; `compiled.refresh(m)` rebuilds a new
global model. Use `compiled.validate_original(m, r.x)` to check the source model.
Nonlinear equalities and indicators are checked in original coordinates.

SCIP's global bound, gap, nodes and version are retained separately in raw
statistics. `solve_verified()` rejects this backend's solver-reported global
result because no independent generic global proof is reconstructed. Relative
gap, time, node and SCIP memory limits, progress callbacks and cooperative
cancellation are supported. Memory is SCIP's accounting limit, not a process
RSS cap. Callback interruption cannot preempt every native operation; model
construction and cleanup may add latency around a time budget.

## Generalized power cones and conservative conic bounds

```python
from solverpilot.model import Model, GeneralizedPowerCone
from solverpilot.conic import ClarabelBackend

m = Model()
x = m.variable(5, lower=[2, 3, 5, 0, 0], upper=[2, 3, 5, 10, 0])
m.add_in_set(x, GeneralizedPowerCone((0.2, 0.3, 0.5), tail_dimension=2))
m.minimize(-x[3])
r = m.solve(backend=ClarabelBackend())
```

Positive head weights sum to one; the geometric product bounds the Euclidean
norm of the tail. Weights are normalized only within the accepted 1e-12 input
sum tolerance, frozen, included in hashes and passed to Clarabel. Validation
uses a scaled log-domain calculation, checks nonnegative heads and handles the
zero boundary explicitly. It remains tolerance-qualified numerical validation.

Clarabel duals are reconstructed into original linear, SOC, RSOC and PSD
coordinates. Small inward perturbations are permitted only when the resulting
dual is independently rechecked. SOC/RSOC dual membership uses exact rational
inequalities; PSD uses the conservative PSD certificate engine. An exact
original-domain residual correction yields the lower bound. `solve_verified`
can accept a sufficiently small gap with a validated primal on this scoped path.

Unbounded-domain residuals, unsupported PSD certificates, or exponential/power
dual membership return unverified. A solver's `Solved`/`AlmostSolved` message
does not override those checks. Bounds refer to the represented binary64 model;
the final primal feasibility and gap remain tolerance-qualified.

## CPU PDLP and certificate recovery

```python
from solverpilot import solve
from solverpilot.backends.pdlp import PDLPBackend
from solverpilot.validate import ValidationTolerances

r = solve(p_lp, backend=PDLPBackend(time_limit_s=20),
          tolerances=ValidationTolerances(feasibility=1e-7))
```

PDLP accepts continuous LP and nonnegative diagonal convex QP. Off-diagonal
quadratics and integer variables are rejected. Solver-reported corrected dual
bounds remain labeled as such; the common LP/QP checker independently processes
canonical duals where possible. Process startup and data transfer count toward
the PDLP adapter's wall budget. This CPU integration makes no GPU speed claim.
PDLP dual infeasibility maps to `infeasible_or_unbounded`: a primal feasible
origin is additionally required to establish unboundedness. This distinction
follows the [upstream termination contract](https://github.com/google/or-tools/blob/stable/ortools/pdlp/solve_log.proto).

```python
from solverpilot import solve
from solverpilot.validate import recover_lp_certificate, verify_lp_certificate

r = solve(p_lp, certificate_recovery=2.0)  # extra recovery ceiling, seconds
c = recover_lp_certificate(p_lp, termination="infeasible", time_limit_s=2.0)
verified = verify_lp_certificate(p_lp, c)
```

Recovery constructs an auxiliary Farkas LP, or a feasible origin and recession
ray LP. Every returned witness is checked against the original model. A saved
`verified=True` field is insufficient; use `verify_lp_certificate` to recheck.
Recovery is continuous LP only and limited to 10,000 rows plus variables and
1,000,000 nonzeros. Within high-level `solve`, recovery uses the smaller of its
ceiling and the remaining common wall budget. No witness yields `unknown`.

## Reuse and learned suggestions

```python
from solverpilot.runtime.reuse import reuse_evidence
e = reuse_evidence(r)
print(e.workspace, e.primal_dual_start, e.numeric_factorization)
```

OSQP vector-only edits update `q/l/u` and omit `Px/Ax`; matrix edits still update
the matrices. `automatic_enabled` means starts are enabled, not that a particular
start or factorization was actually reused. Hidden factorization remains
`unknown`. Clarabel reports observed workspace reuse only when its update ran.

`solverpilot.intelligence.shadow.observe_shadow_policy` accepts a caller-owned
predictor and immutable pre-solve feature record, records feature/prediction cost,
and checks a supplied eligible backend set. It never changes the production
backend. Predictor failures become failed observations. Run costly predictors
outside solve budgets; this synchronous helper is not a sandbox or timeout.
Use the existing benchmark policy evaluator for paired, overhead-inclusive,
family-separated held-out comparison. There is no automatic promotion switch.

## Exact SCIP and GPU readiness

`solverpilot.runtime.integrations.integration_readiness()` inspects exact-mode
and certificate parameters in the installed SCIP build and basic cuOpt platform
prerequisites. It deliberately reports **qualified=False** for these future
integrations. Standard PySCIPOpt wheels tested here did not expose exact mode;
an exact MILP build plus independent certificate verification is still required.
cuOpt requires a compatible Linux/WSL2 CUDA environment and a measured execution
adapter qualification. Detecting a package or GPU is not qualification.

Neither an exact SCIP execution adapter nor a cuOpt GPU execution adapter is
shipped in 0.4. Generic exact MINLP, global optimality of arbitrary functions,
and universal infeasibility/unboundedness certificates remain outside scope.

## Reproducing validation

The `test_04_*` regressions include the near-negative counterexample, PSD
congruence checks, analytic cone optima, 24 nonconvex integer quadratics compared
with exhaustive enumeration, nonlinear equalities, nonsmooth expressions,
indicator bodies, callback failures, process budgets and certificate tampering.
`benchmarks/upgrade04_regression.py` measures serial total call time on small
synthetic workloads, including startup/validation, without a general speed claim.

Release qualification tests the exact wheel across Python 3.12–3.14 on Windows,
Linux and macOS and exercises optional backends separately. Consult the workflow
run for the exact artifact; historical evidence never qualifies a new wheel.

Runnable additions: [global](../../examples/22_global_optimization_optional.py),
[generalized power](../../examples/23_generalized_power_optional.py),
[PDLP/certificates](../../examples/24_pdlp_and_certificates_optional.py).
