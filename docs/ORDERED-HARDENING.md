# Ordered hardening after the public LP qualifications

This work follows six priorities in order. Changes are qualified separately from
promotion of a learned model or publication of a new package.

| Step | Acceptance rule | Status |
|---|---|---|
| 1. Verification cost and call deadlines | Exact arithmetic agrees with the Fraction reference; paired timing improves; late isolated results cannot become successes | Implemented; final integration pending |
| 2. Certificate coverage | Additional original-model bounds with adversarial rejection tests; no tolerance relaxation | Implemented; public replay retains an explicit incomplete case |
| 3. Learned routing value | Measured total-cost gain guard, training-only calibration, fresh evaluation; remain opt-in unless promotion passes | Implemented; synthetic development evaluation only, public promotion pending |
| 4. Operational qualification | Repeated solves, resource observations, cancellation/recovery, multiple timing budgets and CI platforms | Local qualification passed; full CI pending |
| 5. Result meaning | Feasibility, proof source and deadline reported separately without breaking legacy status | Pending |
| 6. Distribution and readiness | Installed artifact distinguishes shipped/experimental capabilities; readiness is separate from execution qualification | Pending |

## Step 1: exact arithmetic and deadline accounting

Exact binary64 dot products now accumulate integer numerators aligned to a shared
power-of-two denominator. Fraction reduction happens once per dot product rather
than after every multiply/add. Random extreme exponents, cancellation, subnormal
products and overflow-sized exact products are checked against a separate Fraction
reference. Operand decompositions are reused within each sparse matvec; no global
mutable problem cache was introduced.

The runtime can reject invalid primal/stationarity candidates before expensive
proof arithmetic. Direct `verify_optimality` calls retain full diagnostic gaps by
default; `fast_reject=True` opts into early rejection. No previously verified
candidate is intentionally weakened by this optimization.

Ordinary and production solve calls deduct setup/planning from the native budget
and record `raw_statistics['call_budget']` for the full call. These remain native
soft limits: validation or solver shutdown can overrun them. Optional infeasibility
diagnosis receives only remaining time.

`solverpilot.runtime.deadline.solve_with_deadline` provides an explicit isolated
CPU LP/QP call using existing batch workers. Use a multiprocessing main guard.
Startup, solve, proof checks, transport and parent primal validation count; cleanup
may add latency. A late result is discarded and never relabeled an on-time success.
Only supported in-process CPU adapters are accepted, avoiding nested worker orphans.
The returned `BatchItem` carries original model identity and independently verified
optimality from the owned worker, gated again by parent primal validation.

The paired verifier-only benchmark uses three planted bounded LP sizes and three
alternating repetitions. It is not an industrial or full-solve speed claim.

## Step 2: optional equality-basis certificate recovery

`solverpilot.validate.lp_dual.recover_lp_optimality` can eliminate coupled free
variables using exact linear combinations of original equality rows. Pivot,
nonzero-visit and rational-bit caps bound work. Inequalities never become
equalities, original primal checks still apply, and incomplete recovery returns
an unverified check. This explicit API is separate from ordinary solve costs.

Constructed coupled systems recover exact bounds, including ten seeded integer
systems with known unique solutions. Adversarial primal, dual and resource-cap
tests reject invalid or incomplete evidence. Replay of consumed public cases
keeps unitcal_7 verified but pilot4 still has no finite residual lower bound.
This is a coverage improvement, not a generic certificate algorithm.

## Step 3: total-cost gain guard

`calibrate_gain_guard` consumes training observations with both learned candidates
and the production planner. All repeats count, with PAR10 for failed or late
independent verification. Every calibration group must benefit after routing
overhead, sufficient groups must support a switch, and no verified repeat may
be lost. The guard binds the model, environment, implementation and time budget.
Changed bindings, expensive setup or missing calibrated gain use production.

Pass `gain_guard=guard` to explicit `solve_robust_lp`; the older unguarded research
API remains available. Guard artifacts are digest-checked on load. Neither API
enables automatic routing. Calibration is not a statistical promotion test.

`benchmarks/qualify_gain_guard.py` freezes 24 new synthetic training cases, model
and guard before solving 12 disjoint synthetic evaluation cases, twice each.
This is a development regression, not fresh public/industrial evidence. Old
public cohorts remain consumed. A new public qualification is still required.

## Step 4: lifecycle and resource observations

A 35-call LP plus 35-call QP soak found that native OSQP defaults produced invalid
cold candidates before warm starts improved them. The built-in registry now uses
1e-8 absolute/relative OSQP tolerances and polishing; explicitly constructed
adapters keep their own settings. Independent validation is unchanged. All 70
calls then passed primal and independent optimality checks.

Timeout, active cancellation and worker crash are followed by successful requests
in the same executor. Sequential batches now preserve independent proof metadata.
Cancellation is checked again after parent validation. The CI OS/Python matrix
runs the resource observation script and preserves its JSON.

On the local Windows run, post-warmup parent+child RSS range was below 0.2 MB for
each 35-call series. This short observation proves neither absence of leaks nor
industrial reliability. Deadlines of 10 microseconds and 0.1 seconds returned
timeouts with no candidate; 5 seconds returned a verified result. Process cleanup
can exceed the requested interval, and that overhead is recorded explicitly.
