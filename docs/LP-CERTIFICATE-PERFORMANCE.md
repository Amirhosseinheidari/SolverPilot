# LP certificate cost and the 68 incomplete public checks

> Release inclusion: the implementation described here is included in 0.4.1.
> The original development measurements and artifact versions below are historical
> evidence, not measurements of the new 0.4.1 distribution.

Unreleased development after 0.4, based on main commit `43118b4`. The earlier
[repeated-LP qualification](REPEATED-LP-QUALIFICATION.md) remains a historical
record, not a measurement of this implementation. No version bump or learned
routing promotion is part of this change.

## Cause and correction

Profiling the current baseline, rather than assuming the October 13 regression
still had the same cause, identifies exact implied-box propagation as the main
public-case certificate cost. STOCFOR1, BRANDY, GANGES and DEGEN3 spend roughly
90% or more of their instrumented certificate/run time there. Repeated Fraction
construction, product addition, division and comparison dominate. Instrumented
profiles explain attribution; they are not the performance comparison.

Propagation now multiplies binary64 dyadics as integers and aligns their powers
of two before summation. Only the final quotient needs a Fraction. Cached row
coefficients and endpoint representations are local to this invocation; bounds
are still derived from the original rows and rounded outward. The bounded visit
order and three-pass limit remain. Empty rows are skipped before allocating row
caches. A separate Fraction implementation and adversarial cancellation, scale,
contradiction and partial-propagation tests check equivalence.

For an LP with canonical minimization cost `s*c`, row duals `y_r` and bound duals
`y_b`, the lower bound now uses

```
q = s*c + A.T*y_r
L = -y_r*b_r + min(q*x over the original or conservatively implied variable box)
```

Variable bounds are enforced by the box, so they need not also appear in this
Lagrangian. For every feasible point, `y_b*(x-b_b) <= 0`; omitting that term
therefore strengthens a valid lower bound. Previously an overshooting rounded
bound multiplier could turn a tiny residual negative on a variable with no upper
bound and make the correction infinite. Full supplied dual signs, stationarity,
complementarity and original primal checks remain mandatory. No small residual
is rounded to zero and no tolerance is increased. QP certificate formulas remain
unchanged. LPs also avoid the quadratic path's identically zero exact products.

If original/implied boxes and greedy row substitution cannot bound the residual,
the standard LP path now tries exact equality elimination capped at 256 pivots,
50,000 nonzero visits and 8,192 rational bits. Only rows with exactly equal finite
endpoints qualify as equalities. Limits return an unverified result; they never
create a certificate. Shared `ProofPolicy` time/work accounting includes row
loading, pivots and fill-in. Its final boundary rejects late success. These are
cooperative controls, not a hard process deadline. Larger and inequality-slack
recovery remain explicit opt-ins. Their residual/base/slack penalties consistently
use zero variable-bound multipliers when starting from this row-only Lagrangian.

## Replay and comparison protocol

The frozen eight-model Netlib admission and all 256 intended observations are
retained. The fixed-candidate replay separately loads the original October 13
candidate/dual vectors, checks them against the original arrays/reference, then
runs the public verifier. It preserves illegal-bound-dual preparation followed by
full rechecking, missing data, exceptions, coverage gains and losses.

The 68 historical failures comprise 32 STOCFOR1, 32 BRANDY and four FFFFF800
observations. Row-only bounds resolve STOCFOR1's bound-multiplier cancellation;
the other 36 need equality combinations beyond greedy row substitution. These
are related transformed/repeated observations, not 68 independent new problems.
Verification means a tolerance-qualified numerical optimum with a conservative
bound, not an exact feasible rational primal or a generic exact certificate.

The final installed-wheel replay independently rechecks all 256 original vectors:
the current baseline verifies 188, and the candidate verifies 256, with 68 gains
and no losses. All original-coordinate feasibility/objective/reference checks
also pass. The recovered observations break down as follows:

| Family | Recovered observations | Recovery | Maximum reported gap |
| --- | ---: | --- | ---: |
| STOCFOR1 | 32 | Row/domain lower bound | 3.664e-10 |
| BRANDY | 32 | Equality elimination, 25-26 pivots | 9.730e-12 |
| FFFFF800 | 4 | Equality elimination, 9 pivots | 2.002e-6 |

The maximum equality work is 5,660 nonzero visits, below the 50,000 cap. None of
these recoveries uses the extended inequality-slack path. Gaps have the objective's
units and should not be compared across families without scaling.

`benchmarks/replay_lp_certificates.py` provides a new fixed-candidate protocol.
Its `replay_s` includes original-data checking, optional dual preparation/retry,
public validation and serialization, while model construction is outside. It
must not be compared directly to historical trace `certificate_s`.

## Regression validation

The final local suite passes 2,334 tests with zero failures and 49 skips in 244.40
seconds on Python 3.12.14. Line coverage is 86.28%; branch coverage is 72.49%.
Skipped tests require unavailable bundled C APIs, external exact/GPU evidence or
executables, symlink support, or POSIX process groups. They are not passing tests.
Correctness-focused Ruff rules and the configured six-file mypy scope pass.

New regressions cover exact propagation equivalence, extreme dyadic scales,
partial work budgets, lazy preparation on wide/empty models, bound-dual overshoot,
min/max objectives and offsets, wrong candidates/duals, genuinely unbounded tiny
objectives, coupled equalities, rejection of nearly equal inequality endpoints,
late/budget-exhausted recovery, explicit slack recovery consistency, and missing
historical candidate records.

## Installed-wheel measurements

The [machine-readable summary](evidence/lp-certificate-performance/summary.json)
retains source/wheel/input hashes, every case ratio, paired process drift, proof
counts, phase times and skipped-test reasons. Both wheels use Python 3.12.14,
NumPy 2.5.3, SciPy 1.18.1 and the same HiGHS 1.15.1 native extension on Windows 11.
Threads are pinned to one. No source tests or other task benchmarks ran during
the serial campaign; unrelated OS background activity was not controlled.

The campaign contains 1,024 public solves, 3,072 measured synthetic solves plus
768 warmups, and a 480-solve analytic stress challenge. All intended observations
were retained and passed their respective gates. Both candidate public runs
verify 256/256; both baseline runs verify 188/256. Each public run records exactly
96 native updates, without isolated-thread fallbacks. All measured synthetic
observations have independent numerical evidence. Synthetic sequences still use
stateless native solves; they must not be described as native reuse.

Times below are pooled medians in milliseconds. The final column is the geometric
mean of **per-case candidate/baseline median ratios**, so below one is faster.
It is not the ratio of the two pooled medians. A and B ran in A-B-B-A order in
fresh processes; two repetitions per version support descriptive observations,
not statistical significance or a universal performance claim.

| Public workload | Presolve | Total before -> after (ms) | Certificate before -> after (ms) | Case ratio |
| --- | --- | ---: | ---: | ---: |
| Cold, no native reuse | off | 184.34 -> 89.81 | 173.95 -> 78.11 | 0.492 |
| Cold, no native reuse | on | 184.09 -> 88.34 | 172.70 -> 70.42 | 0.484 |
| Retained sequence initial | off | 178.53 -> 88.82 | 168.47 -> 77.47 | 0.497 |
| Retained sequence initial | on | 186.30 -> 79.92 | 175.30 -> 68.18 | 0.493 |
| Native sequence update | off | 185.62 -> 82.95 | 180.54 -> 76.65 | 0.430 |
| Native sequence update | on | 178.64 -> 73.57 | 173.70 -> 68.69 | 0.445 |

Public total time includes construction/update, execute and the independent
original-data check; first calls are retained. Candidate runs also establish more
proofs, so this compares the complete implementations with differing coverage.
The summed certificate/total cost share falls from 84.7-86.0% to 66.3-68.3% for
cold public solves. It remains about **90.8%** for native updates, down from
96.7-96.8%: verification is still the dominant cost when native solving is cheap.

| Synthetic stateless workload | Presolve | Total before -> after (ms) | Case ratio |
| --- | --- | ---: | ---: |
| Cold | off | 7.971 -> 7.451 | 0.960 |
| Cold | on | 8.586 -> 7.697 | 0.950 |
| Sequence initial | off | 7.764 -> 7.312 | 0.955 |
| Sequence initial | on | 8.437 -> 7.264 | 0.928 |
| Sequence update, stateless native | off | 8.038 -> 7.148 | 0.957 |
| Sequence update, stateless native | on | 8.268 -> 7.790 | 0.936 |

### Regressions and uncertainty retained

The primary campaign is not uniformly faster. SEBA is 10.6% slower in the
presolve-off sequence initial group and 12.3% slower for presolve-on cold solves
(about 2.5-2.9 ms). The two `network-32` cases regress by 7.2-26.4% with
presolve off, depending on phase (about 0.3-1.0 ms). The 32-variable scaled
production case regresses by 1.9-2.5% in five groups (about 0.06-0.08 ms).
Every case ratio, including these regressions, remains in the summary.

Phase inspection also shows variation in unchanged native construction/solve and
primal-validation stages. A separately frozen, post-campaign diagnostic therefore
replays fixed vectors for these four cases in two A-B-B-A blocks: 1,600 measured
verifications, plus 160 warmups. All pass. Pooled verifier-only candidate/baseline
ratios are 0.909 and 0.942 for the network cases, 0.932 for production, and 0.689
for SEBA. Individual process medians vary substantially. This does not establish
a persistent verifier slowdown, but neither replaces nor erases the primary
end-to-end regressions. Larger controlled-host runs are needed for small-effect
claims.

The analytic challenge passes all 240 cold/reuse pairs, including changed active
sets, bounds, objectives, rebuilds, infeasible/unbounded cases and tiny limits.
Each mode has 192 numerical optimality verifications and 48 cases with optimality
not established; expected termination/limit outcomes are not promoted into proofs.
No guarantee of exact rational primal feasibility, arbitrary termination
certificates, factorization reuse, or production-user performance is added.

### Reproduction and artifact identity

The retained evidence package contains the two wheels, original admission and
historical vectors, frozen runners/protocol, complete raw outputs, profiles,
regression logs, audit scripts and a SHA-256 manifest. The candidate wheel SHA-256
is `e9effc216d7d1bff19d3eb13e3666faa728d1a776bfa8d20392835df5cacecea`;
baseline is `ad89aea2aea73a1080c2b669be27b4940ded4003eb54b1248f08e93bb2fdfed8`.
Version remains 0.4; wheel/source hashes distinguish this unreleased work.

Run the replay against the intended installed wheel, with no source checkout on
`PYTHONPATH`, using the archived paths appropriate to the host:

```text
python benchmarks/replay_lp_certificates.py --prepared ADMISSION --previous ORIGINAL_RESULT_JSON --output NEW_JSON --require-all
python benchmarks/compare_lp_workflows.py --preset standard --steps 4 --repeats 4 --warmups 1 --workflows solverpilot --output NEW_JSON
```

Repeat the synthetic command with `--presolve`; use the archived original public
runner for the frozen public admission. The current runner is not substituted
for the runner hash recorded in that admission. The archive scripts record the
full serial comparison and its post-campaign diagnostic; neither modifies the
old scientific evidence.
