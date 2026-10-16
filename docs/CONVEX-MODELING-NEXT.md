# Convex quadratic constraints, piecewise costs and inventory planning

These additions are included in version 0.4.1. The earlier published 0.4 wheel
does not contain them. Their mathematical and execution boundaries remain below.

## Automatic continuous QCQP lowering

```python
from solverpilot.model import Model

m = Model("ball")
x = m.variable(2, lower=-2, upper=2)
m.add((x**2).sum() <= 1, name="unit_ball")
m.minimize(-x.sum())
compiled = m.compile()
result = compiled.solve()  # Clarabel when available
assert compiled.validate_original(m, result.x).valid
# Analytic optimum: -sqrt(2), at [1/sqrt(2), 1/sqrt(2)].
```

Eligible continuous minimization models now recognize quadratic inequalities
before the generic NLP dispatch. Direct affine-square graphs, vector reductions,
dot products, affine right-hand sides and concave `>=` inequalities can lower to
rotated SOCs. Represented positive weights use unit-square epigraphs with linear
weights, avoiding rounded square roots and weight-times-factor products.

A bounded exact rational LDL path also handles small general PSD quadratics.
Negative pivots and nonzero residuals at a zero pivot reject the bridge. Rational
factors may be represented by integer-cleared RSOC maps only when every emitted
coefficient is exactly representable as binary64. No eigenvalue tolerance,
clipping, approximate Cholesky or rounded polynomial expansion establishes the
new bridge's equivalence. Constants mean their represented binary64 rationals.

The automatic scope is deliberately bounded: 4,096 expression elements, 100,000
charged arithmetic/term-copy work units, 2,048-bit fractions, at most 32 active
variables for general LDL and 4,096 emitted square factors. Limits are checked
before graph-term concatenation/output construction. Exhausted or unsupported
bridges retain existing routing. Quadratic equality/two-sided rows, nonconvex
curvature, discrete models, unsupported objectives and inexact emitted maps do
not gain a conic claim. Explicit cone/atom paths keep their existing contracts.

Original variables remain first; `reconstruct_primal()` strips new auxiliaries.
Source maps record generated rows/cones, parameter dependencies and exact
decomposition preconditions. The direct QCQP compilation hash includes concrete
layout/certificate identity. `execution_ir.metadata["semantic_hash"]` describes
the lowered model; `qcqp_source_semantic_hash` and `qcqp_source_data_hash` identify
the bridge input. `CompiledModel` carries its source model's hashes. These names
also distinguish layers when an outer compiler composes transformations.

Parameters trigger full relowering with at most eight owned numeric snapshots.
Changing signs, factorization or layout can change the route and compilation
hash. Stale original-model validation raises; a cache never reuses candidate
validation. Semantic dual/certificate reconstruction remains unavailable in
this first bridge: independent conic bounds concern the execution IR. Original
semantic primal validation remains a separate check.

For these compiled QCQP models, scalar source rows are rechecked with bounded
rational graph/polynomial evaluation; this avoids expanded-float cancellation
such as `(x - 1e8)**2`. Variable domains and other constraint sets retain their
existing validators. Exhausted source checks return invalid explicitly. This is
a primal check with the requested tolerance, not an exact optimality certificate.
An `x0` hint is shape/finiteness checked and warns that this automatic path does
not use it. Use `solverpilot.nlp.compile_nlp_model` explicitly when an NLP initial
point is required.

## Explicit piecewise-linear helpers

```python
from solverpilot.model import Model, piecewise_linear_epigraph, piecewise_linear_graph

m = Model("cost")
quantity = m.variable(lower=0, upper=4)
cost = piecewise_linear_epigraph(m, quantity, [0, 2, 4], [0, 2, 5], name="cost")
m.minimize(cost.output)  # The helper does not change the objective.
compiled = m.compile()
result = compiled.solve()
assert cost.validate_original(compiled, result.x).valid
```

`piecewise_linear_epigraph` projects whole-knot barycentric weights into
`output >= f(input)` for convex data. Slopes are compared as exact rationals;
rounded slopes/intercepts are not emitted into the model. Larger outputs remain
feasible, so equality needs an appropriate minimizing use.

`piecewise_linear_graph` represents `output == f(input)`, including nonconvex
zigzags. Each selected segment has two nonnegative endpoint weights summing to
its binary. Original knot coefficients are used directly; breakpoint differences,
arbitrary Big-M and native SOS2 support are unnecessary. A single segment uses
only continuous weights. This is an exact graph formulation, not a guarantee of
exact arithmetic by a numerical MILP solver.

Inputs must be scalar affine expressions owned by the model. Knots/values are
owned numeric constants, losslessly representable as binary64, with 2–256 finite
strictly increasing knots and matching values. Mutable parameter knots are not
supported; parameters in the affine input use existing refresh semantics. The
knot interval is enforced; no extrapolation occurs. Invalid helper inputs are
rejected before adding variables/constraints. Integer inputs are permitted.

`handle.evaluate(value)` displays adjacent-knot interpolation and rejects points
outside the interval. `handle.validate_original(compiled, candidate)` additionally
checks the source relation using bounded rational affine evaluation/interpolation
and requires lifted semantic feasibility. It detects interpolation errors even
when the displayed float rounds away a difference. A stale snapshot raises;
malformed/nonfinite candidates and exhausted source checks return invalid with a
reason. Small permitted endpoint violations remain recorded. This is a primal
relation check, not an optimality proof or an automatic extension of generic
`CompiledModel.validate_original` to native PWL semantics.

## Complete advance-commitment inventory fixture

```bash
python examples/production_inventory.py
```

The example combines:

- integer batches of two units, two-batch release capacity and setup binaries;
- releases at times 0 and 1, arriving in demand periods 1 and 2;
- scenario demands `[2,2]` and `[2,4]`, with equal probabilities;
- convex PWL production costs at quantities `[0,2,4]` of `[0,2,5]`;
- setup cost 3, holding cost 0.5 including terminal inventory, lost-demand cost 6;
- CVaR at confidence 0.5 followed by expected loss to resolve recourse ties.

All releases are chosen before observing demand. This is an advance commitment
plan, not an adaptive multistage policy. An independent oracle enumerates all
nine release plans and uses direct stock-flow arithmetic, without the modeling
formulation or solver. The unique best releases are `[2,4]`, scenario losses
`[14,13]`, CVaR 14 and expected loss 13.5. Both prioritized objectives are checked,
along with original feasibility and frozen PWL relations. Lexicographic locks
and native solves remain numerical and tolerance-qualified. Linear production
costs `[0,2,4]` provide a separate control with CVaR 13.

The fixture is synthetic and small. It does not establish industrial validation,
participant adoption or generic stochastic inventory performance.

## Reproducible comparison with common numerical proof

```bash
python benchmarks/compare_lp_workflows.py --preset standard --repeats 10 --warmups 1 --steps 4 --max-wall-s 900 --workflows solverpilot,solverpilot-reuse,direct-highspy,cvxpy-highs,pyomo-appsi-highs --common-proof --presolve --output NEW.json
```

Repeat with `--no-presolve` and a new output. The optional v3 mode supplies the
same analytic original-data dual witness to the same numerical LP verifier for
every wrapper's actual candidate. A rival is not claimed to have returned that
witness. Objective-report consistency remains checked separately, and accepted
observations require the common numerical proof. All SolverPilot internal checks
remain enabled. No exact certificate/evidence-export parity is inferred.

`checked_total_s` retains the historical definition; `proof_checked_total_s`
adds fresh common optimality checking and original-data canonical preparation.
The output records separate timings, checker source-file hashes, all candidate
checks/failures, package/native versions, runner/data hashes and warmups. Compare
checker/dependency identities before comparing source versions. Native solving
uses the same HiGHS library; synthetic planted optima, imports/startup and cleanup
exclusions, descriptive p95 and unobserved rival internal proof work retain the
documented limits. This experiment alone authorizes neither general speed claims
nor learned production routing.
