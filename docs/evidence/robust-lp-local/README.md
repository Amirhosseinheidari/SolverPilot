# Public-trained LP routing and certificate recovery — 2026-09-29

**Decision: retain explicit experimental routing; do not enable it automatically.**
The new public-trained route passed the preregistered comparisons against
`solve_production()`, but failed the full promotion rule because it did not beat
ordinary `solve()`. The numerical certificate improvements are separately tested
correctness work, not evidence that the learned selector is universally faster.

## What was implemented

- Support/environment/availability abstention now uses the conservative production
  planner in the new explicit `solve_robust_lp` wrapper, with one remaining budget.
- Training now includes public Netlib LPs and MIPLIB continuous relaxations.
- Illegal infinite-bound dual multipliers can be projected and independently
  rechecked. Exact implied-bound propagation and original-row residual
  substitutions recover some previously unavailable LP lower bounds. Raw solver
  duals remain preserved; incomplete recovery remains unverified.

The [implementation and reproduction guide](../../ROBUST-LP.md) explains the
arithmetic, work caps, API, timing profile and limits. Original stage3/stage4 model
artifacts and outcome files were not rewritten.

## Frozen experiment and data separation

Source was frozen at `2dca4c10218c57aa15aaf55b65decb7102c4356d` before measured
training. [protocol.json](protocol.json) records its source hashes, environment,
budgets, candidate set and acceptance rule.

There were **456 planned calls**: 144 training calls on the 24 consumed stage4
public cases, 288 held-out calls on 24 fresh cases, and 24 separate stress calls.
No timed training or held-out outcome was replaced or discarded. No worker-process
error or controller timeout occurred in either measured cohort. Runtime `error`
statuses are retained; they are not worker crashes.

The fresh cohort has 12 Netlib LPs and 12 MIPLIB continuous relaxations, 24 distinct
declared groups and 24 distinct canonical hashes. Its admission audit excludes
207 known names and related families. Training/test names, groups and model hashes
are disjoint; two MPS readers agreed on every admitted model. One official download
was rejected before solving for inconsistent variable bounds; it remains in the
exclusion record. Raw third-party MPS files are not redistributed. This is an audit
of recorded history, not proof that unrecorded outside datasets were never seen.

- [Training cohort](training-cohort.json) and [all 144 training outcomes](training-observations.json)
- [Fresh held-out admission](heldout-cohort.json) and [all 288 held-out outcomes](heldout-observations.json)
- [Training rows](training-rows.json), [model](model.json) and [pre-test model freeze](model-freeze.json)

The model payload SHA256 is
`e6cde2ac35d850820e5eb89c8ed1b11a64ba1068bdcb0cdfd93536febad71d96`.
Its public-training baseline is HiGHS simplex. A single split on log variable
count selects HiGHS IPM on the left and simplex on the right; support checks and
production fallback still apply. It was frozen before the first held-out solve.
No tuning, retraining or support expansion followed held-out results.

The profile is serial fresh processes, one logical CPU affinity, one-thread BLAS,
on the same i7-12650H laptop under Ubuntu 24.04/WSL. API budget is 2 seconds and
the process-tree deadline is 12 seconds. WSL guest UTC is ahead of the host;
original timestamps are preserved and durations use a monotonic clock.

## Held-out results

Each strategy has 48 calls. Backend claims, primal validation and independent
optimality are different quantities. See [outcome-breakdown.json](outcome-breakdown.json)
and [summary.json](summary.json) for machine-readable results.

| Strategy | Backend claimed optimal | Primal valid | Independently optimal, including late | Independently optimal within budget | Mean PAR10 cost |
|---|---:|---:|---:|---:|---:|
| Public-trained robust route | 44 | 44 | 34 | 28 | 51.748 s |
| Ordinary `solve()` | 44 | 44 | 34 | 29 | 49.417 s |
| Ordinary `solve_production()` | 44 | 44 | 32 | 24 | 61.445 s |
| HiGHS simplex | 43 | 43 | 34 | 27 | 54.118 s |
| HiGHS IPM | 40 | 40 | 30 | 26 | 56.490 s |
| CPU PDLP | 20 | 18 | 10 | 10 | 95.672 s |

**PAR10 is penalized completion cost, not average latency.** A successful repeat
costs full process wall time, including imports/parsing. Every unverified, failed
or late repeat costs 120 seconds. API time includes model loading, environment
binding, availability checks, routing and verification; both clocks are retained.

Against `solve()`, the robust cost ratio was **1.0472**, bootstrap 95% interval
**[0.9974, 1.1769]**. Gain, confidence and no-lost-solve gates failed. The lost
within-budget repeat was `sp98ar`: robust results were independently verified but
took 2.050/2.075 seconds, while one default repeat took 1.984 seconds. This shows
sensitivity near the fixed deadline, not proof of a universal slowdown. The
threshold was not relaxed after seeing that result.

Against `solve_production()`, the ratio was **0.8422**, interval
**[0.6303, 0.9992]**; all its gain/tail/no-lost-solve gates passed under this profile.
The full rule requires both default comparisons, so **the overall gate failed**.
The bootstrap uses 2,000 paired group resamples with the preregistered seed.

There were 24 in-support training-baseline decisions, **8 learned IPM switches**,
and **16 out-of-support decisions routed to production**. Unlike the earlier
synthetic model, this policy exercised learned choices on the fresh public set.
No independently verified objective disagreement occurred, including available
Netlib reference comparisons. Integer MIPLIB optima were not used as LP references.

These are new cases and new verification code: do not interpret a comparison with
stage4's 2/48 policy count as a controlled before/after speedup. The table above is
the simultaneous comparison on this fresh cohort.

## Stress and diagnostic evidence

[stress.json](stress.json) contains 24 calls excluded from performance scoring:

- All routes reported infeasible on the infeasible case; none claimed independent
  optimality. Independent infeasibility-certificate recovery was not requested.
- The robust/default/HiGHS routes reported unbounded; PDLP preserved the ambiguous
  `infeasible_or_unbounded` status. No independent optimum was fabricated.
- Robust/default/HiGHS routes independently verified the ill-scaled case; PDLP did
  not return a valid solution within the short budget.
- The tiny-budget robust call stopped at setup exhaustion. All six routes remained
  unsuccessful/unverified on that stress case.

Two preserved development diagnostic rounds are separate from fresh performance
evidence: [development round](development-diagnostics.json) and
[frozen-code replay](frozen-code-diagnostics.json). The original local replay
script is [diagnostic-runner.py](diagnostic-runner.py); its workspace paths are
audit context, not a portable CLI.

On `unitcal_7`, the frozen-code replay independently verified the repaired
certificate, with corrected gap about **1.254e-9**, where the stage4 checker had
rejected an infinite-bound multiplier. `pilot4` still lacked a finite residual
lower bound, so remained unverified despite its primal-valid point. PDLP's
`pilot4` replay varied between a rejected candidate and a no-solution limit across
development rounds. No complete repair of generic LP certificates is claimed.

## Limits and retained decisions

This is a bounded-budget, single-host public experiment, not an industrial
workload census. Only two timing repetitions per strategy were used. This fresh
cohort is now consumed. Further policy or threshold changes require new held-out
evidence; it cannot become an unused test again by being renamed.

The new wrapper remains explicitly experimental. Ordinary routing stays
conservative. LP proof checks retain their tolerances and fail closed on incomplete
recovery. No automatic learned/GPU routing, release tag, version bump or PyPI
publication was enabled. Checksums in [manifest.json](manifest.json) bind captured
files; they are integrity checks, not external attestation.
