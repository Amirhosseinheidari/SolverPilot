# Semantic contracts and verification

This is unreleased development after 0.4. No new PyPI version is implied by the
checkout's version string. The historical published API remains available; new
controls are optional keywords and features use submodule namespaces.

## Correctness repairs

The October audit reproduced twelve defect groups. Regression tests now cover:

1. Symbolic powers accept integers, including NumPy integers, and reject Boolean,
   fractional, string and nonfinite exponents instead of truncating them.
2. `CompiledModel.validate_original` requires the original semantic/data snapshot.
   Refresh after parameter changes; unrelated models are rejected.
3. CasADi lowering uses NumPy row-major indexing and broadcasting for scalars,
   vectors and matrices. Both vector orientations and all four 1D/2D matrix
   multiplication combinations match independent NumPy oracles. `np.sum`
   handles positional integer axes and `keepdims`; unsupported options, including
   tuple axes, raise explicitly. Empty slices are tested, and empty nonlinear
   constraint blocks validate as vacuous constraints.
4. Local NLP maximization negates the backend objective and its stationarity
   gradient, then reports the original objective. It establishes no global proof.
5. Binary enumeration preserves original bounds and validates incumbents before
   replacing them. OA also excludes inadmissible rounded assignments and tightens
   master bounds to actually admissible binary values.
6. Short/nonfinite MINLP candidates produce invalid reports with discrete-variable
   diagnostics rather than indexing errors or a zero integrality violation.
7. Owned result arrays protect values, shape and dtype. `.copy()` returns an
   independent mutable array; pickling retains protected metadata.
8. Trace parameters/fingerprints recursively snapshot nested containers.
9. Portfolio overrun accounting includes the final solve and validation, while
   retaining a valid result even if its native soft limit was exceeded.
10. Power/generalized-power membership uses the conservative original-domain
    comparison for both zero and positive tolerances. Exact extreme boundaries
    no longer become invalid when tolerance is increased.
11. CP domains, coefficients, constraints and candidate assignments are checked
    for lossless finite integrality. Invalid fractional objectives are rejected.
12. TSP brute force/Held–Karp and the small VRP reference compare common-denominator
    integer costs. Float rounding cannot create a false ordering tie. Their exact
    claims concern represented binary64 costs and existing bounded reference scope.

`tests/test_semantic_contract_repairs.py` contains independent analytic and NumPy
oracles. `tests/test_staged_extensions.py` includes adversarial regressions found
by a second independent review of these changes.

## Explicit independent numerical proof policy

```python
from solverpilot.runtime import solve
from solverpilot.validate.workspace import (
    ProofPolicy, ProofRequirementNotMetError, VerificationWorkspace,
)

workspace = VerificationWorkspace(max_entries=8, max_coefficient_terms=65536)
try:
    result = solve(problem, proof_policy=ProofPolicy("required"),
                   verification_workspace=workspace)
except ProofRequirementNotMetError as error:
    result = error.result  # The completed result and its candidate are retained.
```

`standard` preserves existing behavior. `budgeted` requires a positive
`time_limit_s` or `max_terms`; exhaustion cannot promote a proof.
`required` rejects an unmet independent numerical proof requirement after
authorized recovery. `certificate_recovery` can therefore recover an LP
termination witness before required mode is enforced. Work/time limits apply to
optimality checks and termination witnesses/recovery; retries share accounting.

These are cooperative verification limits, including conservative arithmetic
work counting. A single native/integer operation cannot be interrupted by them.
Use process isolation for hard deadlines. `VALID_OPTIMAL` remains backend-reported
optimality plus validated primal; independent evidence stays a separate field.
This API does not replace the separately qualified exact SCIP/VIPR API.

The workspace binds preparation to exact CSR bytes, including duplicate ordering.
It caches coefficient dyadics, never candidate validation or certificate acceptance.
LRU entries, coefficient storage and per-entry row-pointer metadata are capped;
oversized or mostly-empty large-row matrices bypass the
cache. Every primal, multiplier, objective and bound check remains fresh.
`ReoptimizationSession` owns and clears a bounded workspace. The paired
`benchmarks/qualify_verification_workspace.py` experiment includes fresh primal
validation and first preparation. Its generated instances are not public or
industrial qualification, and timing differences are observations, not CI speed
thresholds or routing authorization.

## Aggregate execution budgets and reports

`solve_lexicographic(..., budget=SolveBudget(wall_time_s=...))` spends one allowance
across stages and retains completed stages on exhaustion. Unfinished objectives
are not reported as optimized. `budget_report` includes final elapsed time.

`solve_outer_approximation` and `solve_binary_enumeration` accept a common wall time
budget. Remaining time is mapped before each NLP/MILP dispatch. Expiration retains
the best original-domain-valid incumbent and completed history; unvisited
assignments cannot establish exhaustive optimality. MINLP memory/thread/callback
controls still raise explicitly.

`scenario_sweep(..., aggregate_budget=...)` adds a cumulative execution allowance,
including session setup. Consumer idle time between yields is excluded. Per-scenario
options remain separate and are intersected with the aggregate allowance. Each
yield includes its aggregate budget report.

Native time limits remain soft, including setup, validation and cleanup in total
accounting. Completed mathematical evidence is not erased merely by an overrun.
`solverpilot.reporting.explain_any` gives a common evidence-preserving explanation
for core and specialized results. Specialized numerical results with owned model
hashes can export summary manifests. Full model replay remains canonical LP/QP.

## Convex QP sensitivity and parameterized solves

Existing compiler parameter caches, partial recompilation and OSQP same-sparsity
updates remain in place. `solverpilot.sensitivity.differentiate_qp` adds local
directional derivatives for changes in `P`, `c`, linear `A` and bound endpoints.
It rechecks the original candidate/multipliers and requires a stable active set,
strict complementarity, independent numerical convex KKT evidence, independent
active rows and a well-conditioned nonsingular KKT system. Dense work is capped.

The returned `dx` is conditional on those regularity assumptions and unchanged
activity. Weak activity, redundant constraints, nonunique or ill-conditioned
systems return `undefined`; no pseudoinverse is presented as a unique derivative.
Endpoint directions follow linear rows then variable bounds. Equality lower and
upper directions must agree. Analytic and finite-difference regressions cover the
defined scope. `solverpilot.model.parametric.parameter_data_direction` analytically
differentiates the polynomial LP/QP lowering algebra without perturbing or mutating
the model. Its owned sparse coefficient/bound/offset directions bind to the current
model and execution hashes, under an expression-element cap.
`differentiate_parameter_qp` chains them through regular local QP KKT. Target,
coefficient and right-hand-side cases have independent analytic tests. Bridged,
convex-atom/nonlinear parameter paths and PyTorch wrappers remain future extensions.
OSQP's upstream adjoint API is a separate native
capability, not a general differentiability guarantee: [official API](https://osqp.org/docs/interfaces/C.html).

## Vector convex OA and elementary domains

Componentwise affine squares/even powers, indexing, transpose and nonnegative
sum reductions now propagate certified curvature. Vector nonlinear constraints
produce one supporting cut per component. Uncertified nonlinear products and
nonlinear equalities remain outside this convex binary OA proof path; the explicit
bounded global path is separate. Solver-certified scoped proof is not independent
global verification of the whole OA chain.

Global domain lowering can enclose `exp`, `log` and `sqrt` with rational bounds.
Exp uses range reduction and Taylor remainder bounds; log uses an atanh series
with a geometric remainder; sqrt uses integer square roots. Dyadic outward
rounding and work/range limits keep this bounded. Unsupported/exhausted enclosures
return unknown. Structural positivity handles `log(exp(x))` even below the dyadic
grid, after recursively validating child domains. Ordinary libm plus one
`nextafter` is not used as an interval proof. Global abs/min/max models can clone.

## Sparse PSD and execution profiles

Exact Gershgorin/small rational LDL checks extend to disconnected sparse components
under a total component-work bound. Large block-diagonal, singular PSD matrices
can now be certified without a global dense matrix. One indefinite component
rejects PSD; an oversized/unresolved component retains unknown.

`certify_psd_factor(P, R, weights=d, diagonal=s)` independently checks exact
represented-data equality `P = R.T diag(d) R + diag(s)` with nonnegative `d,s`,
under product/fill/bit limits. Rounded `R.T @ R` is not assumed equal and can be
rejected. `verify_optimality(..., curvature_witness={"factor": R, ...})` can use
this explicit witness for connected matrices beyond the small LDL scope.

`ClarabelBackend(execution_profile="chordal")` keeps native chordal decomposition
for cold solves. `execution_profile="repeated"` enables owned data updates with
presolve/chordal decomposition disabled. Combining `reuse=True` with `chordal`
raises. Settings are recorded. This exposes upstream tradeoffs; it does not claim
a new chordal algorithm or generic exact PSD certification. See [Clarabel data
updates](https://clarabel.org/v0.11/user_guide_data_updating/) and [chordal
decomposition](https://clarabel.org/stable/user_guide_chordal/).

## Risk and uncertainty

`solverpilot.model.robust.minimize_cvar` adds probability-weighted finite-scenario
upper-tail CVaR. Confidence is in `[0,1)`; one means use worst-case instead.
Probabilities must be finite, nonnegative, match the scenario count and sum to
one within `1e-12`; they are not silently normalized. Confidence zero lowers
directly to expected loss, avoiding a free threshold under rounding. Tiny positive
confidence that cannot distinguish the represented tail weights raises explicitly.
Zero-probability scenarios, ties and nonuniform weights are permitted.

`ellipsoidal_leq` models coefficient uncertainty `{a + F u : ||u||2 <= radius}`
with `a.x + radius*||F.T x||2 <= bound`. Rectangular/rank-deficient factors and zero
radius are supported. Existing box and finite worst-case helpers remain available.

## Public and operating qualification

`benchmarks/materialize_lp_relaxation.py` creates an actual continuous MPS from a
mixed-integer source, checks canonical reader/writer agreement, hashes source and
materialized models, and builds a separately checked LP reference. An integer
optimum is never an LP-relaxation reference. Unverified references remain null;
the tool grants no production authorization.

Existing public-family novelty/consumed-holdout gates remain enforced. A fresh
manifest still needs audited independent families, frozen training/calibration/test
splits, independently checked objectives and paired setup/proof-inclusive outcomes.
Actual participant/operating observations and compatible GPU execution evidence
cannot be manufactured from unit tests. Automatic learned routing remains disabled.
General sparse PSD without a verified factor and broader independent generic
global certificates require further
work; these changes do not claim those capabilities.
