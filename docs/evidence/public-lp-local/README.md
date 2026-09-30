# Public LP qualification — 2026-09-29

**Decision: do not enable the current learned policy in production.** The frozen
synthetic-trained model failed this public-workload gate. Its earlier synthetic
improvement did not transfer to this cohort or execution profile.

The experiment used the same Intel Core i7-12650H laptop, Ubuntu 24.04 under WSL,
and CPU solvers. There were **288 public calls** (24 instances × 6 strategies × 2
repetitions), **24 separate stress calls**, and **3 post-outcome diagnostic calls**.
No public outcome was discarded or replaced. There were no controller timeouts
or worker-process errors in the public campaign; public `error` statuses below
are runtime outcomes, not Python-worker crashes.

## Frozen scope and admission

The model is the unchanged [local-experiment model](../learned-lp-local/model.json),
payload SHA256 `90333470414abd395b1a688006a5dc691258c99e160af394af34c8a9504806f2`.
Execution code was frozen at commit `1415f2c048d5e3aaf4df5b994bcaff742b3ace4a`.
[protocol.json](protocol.json) records code hashes, package versions, environment,
CPU affinity, timing budgets and the acceptance rule before solving.

The admitted cohort contains 12 Netlib LPs and 12 MIPLIB continuous relaxations,
with 24 distinct declared groups and model hashes. Known prior instances and
related groups were excluded before outcome-independent hash-order selection.
SolverPilot and HiGHS readers agreed exactly on every admitted model's matrix,
objective and bounds. [cohort.json](cohort.json) preserves URLs, byte hashes,
dimensions, audited prior sources and all exclusions. Raw third-party MPS files
are not redistributed. MIPLIB integer optima were never used as LP references.

WSL guest UTC was about one day ahead of the host; timestamps are retained and
durations use `perf_counter`. Guest RAM differed by 8 KiB from training. The
pre-run environment-binding repair records that difference and both raw IDs;
all observed thread settings and other fingerprint fields matched. This does
not equate the cold, single-CPU-affinity public profile with warm synthetic runs.

## Results: feasibility is different from independent optimality

Each strategy has 48 public executions. All times and statuses remain in
[observations.json](observations.json); aggregated counts are independently
reproducible from [outcome-breakdown.json](outcome-breakdown.json).

| Strategy | Backend claimed optimal | Primal valid | Independently optimal, including late results | Independently optimal within budget | Mean PAR10 cost |
|---|---:|---:|---:|---:|---:|
| Frozen learned policy | 20 | 15 | 4 | 2 | 115.089 s |
| Default `solve()` | 44 | 44 | 18 | 18 | 76.218 s |
| Default `solve_production()` | 44 | 44 | 18 | 18 | 76.270 s |
| Native HiGHS simplex | 44 | 44 | 18 | 18 | 76.227 s |
| Native HiGHS IPM | 42 | 42 | 18 | 12 | 90.906 s |
| CPU PDLP | 20 | 15 | 4 | 2 | 115.096 s |

**PAR10 is a penalized cost, not average solve latency.** Success requires original
space independently verified optimality, <=2 seconds of API work and <=12 seconds
of full controller wall time. Successful repeats cost their full process wall
time, including imports and parsing; every other repeat costs 120 seconds. All
repetitions count, including unverified-but-primal-valid and verified-but-late
outcomes. The two underlying time measurements are retained separately.

The frozen policy/default PAR10 ratio was **1.5100**, with group-bootstrap 95%
interval **[1.1926, 2.1121]**. The policy/production ratio was **1.5090**, interval
**[1.1923, 2.1107]**. These indicate worse penalized completion cost under this
protocol, not a universal claim that every learned solve is 51% slower.
All preregistered gain/tail/no-lost-solve gates against both default APIs failed.
See [summary.json](summary.json) for every comparison and gate.

Every one of the **48 routing decisions abstained as outside training support**
and used the model's training-baseline PDLP. There were zero in-support learned
switches and zero environment-mismatch abstentions. Thus this is a failure of
training coverage and the public suitability of its fallback, not evidence that
a particular learned split made 48 wrong in-support predictions.

No material mismatch occurred among independently verified objective values or
against the available Netlib LP references. The policy nevertheless produced 7
primal-rejected candidates; the validator did not accept them as verified solves.
Default `solve()` produced 44 primal-valid results, but only 18 independently
verified optima: proof availability remains narrower than solver optimality claims.

## Stress results

[stress.json](stress.json) contains all 24 calls, excluded from performance scoring:

- Infeasible: all six routes reported infeasible; none claimed an independently
  verified optimum. Independent infeasibility certificates were not requested.
- Unbounded: HiGHS/default routes reported unbounded; PDLP and its policy fallback
  retained the conservative `infeasible_or_unbounded` status. None claimed verified
  optimality. This preserves the existing ambiguous-status boundary.
- Ill-scaled: both defaults and both native HiGHS configurations returned
  independently verified optima; PDLP and the policy did not return a valid result
  within this short budget.
- Tiny deadline: all six calls remained unsuccessful/unverified. The learned
  route detected setup-budget exhaustion. No late/partial outcome was relabeled
  as a verified success.

## Post-outcome diagnosis, not new performance evidence

Three explicitly posthoc replays are preserved in
[posthoc-diagnostics.json](posthoc-diagnostics.json). The original local diagnostic
script is retained as [posthoc-runner.py](posthoc-runner.py), with its original
workspace paths; it is an audit artifact, not a portable CLI.

- `unitcal_7`: default HiGHS gave a primal-valid point, but the canonical dual had
  a multiplier pointing toward an infinite bound; the independent checker rejected
  the certificate.
- `pilot4`: the production route gave a primal-valid point with stationarity
  residual around 1.14e-13, but unbounded variable directions prevented a finite
  rigorous residual correction. A tiny residual alone is not a global bound.
- Replaying a primal-rejected learned-route case on `pilot4` hit a time limit
  without a solution. That replay did **not** reproduce the original violation;
  no stronger causal claim about that candidate is made.

None of these replays changed the policy, original outcomes, budgets or gate result.
The full qualification cohort is now consumed.

## Development consequences

Keep the ordinary conservative routes and numerical checks. Do not relax proof
requirements or label a backend's status as independent evidence to improve a
benchmark score. Next candidate changes should use the existing production route
as the out-of-support fallback, train on representative public/user families, and
improve dual-candidate preparation/recovery for infinite-bound LPs. Each needs
separate correctness tests and a new untouched performance cohort before promotion.

This is a bounded-budget, single-host public challenge, not an industrial workload
census. Larger time budgets, persistent solver reuse or changed fallback behavior
would be different experiments. No automatic learned routing, GPU routing, version
tag or PyPI publication was enabled by this work.

See [the method and reproduction guide](../../PUBLIC-LP-QUALIFICATION.md).
[manifest.json](manifest.json) binds the captured evidence files; checksums provide
integrity, not external attestation of the measurements.
