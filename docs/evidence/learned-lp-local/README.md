# Local LP selector experiment — 2026-09-29

**Outcome:** the frozen learned selector passed the local validation and held-out
research gates. **Production learned routing remains disabled.** These are fresh
synthetic instances on one laptop, not public/OOD deployment evidence.

The laptop is the same Intel Core i7-12650H system used for GPU qualification.
This experiment uses CPU solvers in Ubuntu 24.04 under WSL. Exact environment,
package versions, thread settings, source hashes and instance identities are in
[protocol.json](protocol.json). Guest UTC was about one day ahead of the Windows
host; raw timestamps are retained, and all durations use the monotonic clock.

## Frozen experiment

- Three generated families: planted sparse ranged LP, separable bounded LP and
  transportation LP; new deterministic seeds, 30 groups, two variants per group.
- 36 training instances, 12 validation instances and 12 initially sealed test
  instances; no group or model-data-hash overlap across splits.
- Three candidate configurations: native HiGHS simplex, native HiGHS IPM and
  isolated CPU OR-Tools PDLP. One solver thread, two repetitions per candidate,
  two-second cutoff, serial order-balanced execution.
- Protocol and source hashes frozen before timing. Fixed warm-ups are separately
  recorded. Each result uses ordinary `solve()` and independent validation.
- Model fitted only on training observations; validation and test predictions
  persisted before outcomes. Test was consumed once, only after validation passed.
- Model: one cost-sensitive split on log(1 + matrix nonzeros). Small cases choose
  HiGHS IPM; larger cases choose PDLP. All candidates, threshold, support ranges and
  training identities are preserved in [model.json](model.json).

## Results

The baseline is CPU PDLP, the single best candidate **selected on training**.
Neither validation nor test outcomes chose this baseline. Times below are mean
PAR10 cost per instance, with feature extraction/selection overhead added to the
policy. Every failed, unverified or late repetition costs 20 seconds; successful
repetitions never hide unsuccessful ones.

| Split | Instances / groups | Frozen baseline | Learned policy | Policy / baseline | 95% group-bootstrap interval | Verified selected repeats |
|---|---:|---:|---:|---:|---:|---:|
| Validation | 12 / 6 | 0.372565 s | 0.090738 s | 0.24355 | [0.08196, 0.47740] | 24 / 24 |
| Held-out test | 12 / 6 | 0.345430 s | 0.072152 s | 0.20888 | [0.07096, 0.40927] | 24 / 24 |

The held-out policy cost was about **79.1% lower than the training-frozen PDLP
baseline** on this cohort. This is not a claim of 79.1% acceleration over
SolverPilot's existing default planner, every CPU solver, or real user workloads.
Eleven of twelve held-out choices differed from the baseline. No selected repeat
lost independently verified optimality compared with that baseline.

There were **360 measured candidate solves** across all splits. Independent
verification succeeded on 108/120 simplex, 112/120 IPM and 120/120 PDLP solves.
The remaining 20 outcomes were reported as `limit_no_solution` and remained in
the penalized costs. Certified objectives were cross-checked across candidates;
two families additionally had analytically planted reference optima.

All predeclared research gates passed on both validation and test: >=6 groups,
>=2 switches, >=3% mean gain, bootstrap upper ratio <1, p90 relative cost <=1.25
and no lost verified solves. Full metrics and gates are in
[summary.json](summary.json); [observations.json](observations.json) preserves every
candidate result, including time limits.

## Limits and next evidence required

This is an offline policy replay against paired measured runs, with measured
feature/decision time added. It is not an end-to-end deployment performance claim.
Model loading, environment capture and availability discovery are amortized setup.
PDLP has a cold child process per solve while native HiGHS uses warm imports;
this reflects the current adapters, not intrinsic algorithm performance alone.

Six held-out groups from three synthetic generators are a small sample. The
feature-range guard cannot recognize all distribution shifts. No threshold or
model was retuned after these outcomes. This test is now consumed; another model
requires a new untouched test cohort.

Before enabling production routing, collect representative public and user LP
families under a frozen protocol, compare against the existing production planner
as well as fixed solver baselines, evaluate actual routed execution and failures,
and repeat on the target deployment environment. Preserve runtime correctness,
capability and fallback checks regardless of model quality. This campaign always
records `production_authorized=false`.

Reproduce with [benchmarks/qualify_learned_lp.py](../../../benchmarks/qualify_learned_lp.py)
and the instructions in [LEARNED-LP.md](../../LEARNED-LP.md). Use a new output
directory. [manifest.json](manifest.json) binds the captured evidence files;
checksums provide integrity, not external attestation of measurements.
