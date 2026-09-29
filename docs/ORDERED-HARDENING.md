# Ordered hardening after the public LP qualifications

This work follows six priorities in order. Changes are qualified separately from
promotion of a learned model or publication of a new package.

| Step | Acceptance rule | Status |
|---|---|---|
| 1. Verification cost and call deadlines | Exact arithmetic agrees with the Fraction reference; paired timing improves; late isolated results cannot become successes | Implemented; final integration pending |
| 2. Certificate coverage | Additional original-model bounds with adversarial rejection tests; no tolerance relaxation | Pending |
| 3. Learned routing value | Measured total-cost gain guard, training-only calibration, fresh evaluation; remain opt-in unless promotion passes | Pending |
| 4. Operational qualification | Repeated solves, resource observations, cancellation/recovery, multiple timing budgets and CI platforms | Pending |
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
