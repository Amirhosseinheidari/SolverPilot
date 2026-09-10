# M11 Status — 0.0.12

## Milestone goal
Make public-corpus benchmarking portable and auditable on a normal network/HPC machine without turning unexecuted MIPLIB/QPLIB work into a fake completion claim.

## Completed in M11
- Added `optimind-bench` portable benchmark CLI.
- Added built-in acquisition specs for MIPLIB 2017 Benchmark v2 and QPLIB current archive.
- Added retrying/resumable `.part` HTTP downloads, SHA-256 acquisition locks and safe ZIP/TAR extraction checks.
- Added `verify-dataset` to re-hash downloaded files, verify MIPLIB manifest/reference counts and check every extracted manifest instance.
- Added deterministic sharding suitable for SLURM arrays.
- Added resumable JSONL execution with truncated-final-record recovery and hard failure for corruption in the middle of a file.
- Run identity now includes protocol fingerprint, direct backend versions and instance SHA-256. Changing solver version/config/instance data produces a new run ID.
- Summary rejects mixed protocol fingerprints and duplicate run IDs.
- Added controller-enforced process isolation / hard timeout, independent of solver-level time limit.
- Added SBS, VBS, PAR-style costs and Dolan-Moré style performance profile summary.
- Added deployable `@auto` evaluation with planner overhead included.
- Added deterministic paired bootstrap 95% interval for policy/SBS ratio.
- Added external frozen-policy evaluation with per-instance selector overhead, so research/ML policies can be evaluated without modifying the production planner.
- Added SLURM array script generation.
- Added result bundles with an internal SHA-256 manifest.
- Added environment/backend-health capture to benchmark artifacts.
- Added M11 HPC runbook and an explicit Benchopt handoff contract.

## M11 harness smoke
- 18 generated LP instances.
- 3 direct algorithm paths + `@auto`.
- 2 repetitions.
- 3 deterministic shards.
- 144 total expected rows.
- Resume re-run skipped all 48 rows belonging to shard 0.
- Hard-timeout isolation smoke: normal isolated solve -> `solved`; intentionally tiny hard timeout -> `hard_timeout`.

### Negative planner evidence
On this short-workload smoke suite:
- direct SBS: `scipy-highs-ds`.
- direct SBS mean cost: ~0.003174 s.
- VBS mean cost: ~0.002890 s.
- deployable `@auto` mean cost: ~0.006308 s.
- policy/SBS point ratio: ~1.987.
- paired bootstrap 95% interval: ~[1.809, 2.257].
- SBS->VBS gap closure: ~-11.01.

This is evidence **against** shipping the current capability-only planner as a performance selector for short LP workloads. No new empirical production rule was added.

## Public data / native-solver boundary
Still not completed in this runtime:
- MIPLIB 2017 benchmark archive execution.
- QPLIB benchmark execution.
- dedicated public `highspy` Python adapter execution.
- dedicated public `osqp` Python adapter execution.
- dedicated public PySCIPOpt execution.
- Benchopt installation/execution.

The MIPLIB metadata-only acquisition attempt still failed because this runtime cannot resolve the external host. This was recorded as a failed external acquisition, not a completed benchmark.

QPLIB in M11 is **acquisition/verification only**. The current canonical QP IR does not yet cover quadratic constraints/discrete QPs, so the QPLIB archive is not labeled runnable.

## Regression gate
Final source-level pytest:
- total: 204
- passed: 201
- skipped: 3
- failures: 0
- errors: 0

Known skips are the dedicated public highspy / OSQP / PySCIPOpt integration modules.

Additional regressions:
- mutation/hash: 1000/1000.
- OSQP translation: 1200/1200.
- MPS semantic cross-check: 200/200, max objective difference ~7.105e-15.
- diagnostics property checks: 200/200.
- native IIS stress: 50/50.
- four-path convex-QP regression: 48 instances / 576 direct calls / zero failures.
- exact brute-force binary MILP regression: 100 instances / zero failures across SciPy-HiGHS, CasADi-HiGHS and CBC.
- research DS/IPM stability run: 24 instances / 240 calls / policy/SBS ~0.922 with bootstrap interval ~[0.876, 0.970].
- HiGHS repeated-LP stateful/cold median ratios: small ~0.637, medium ~0.416, large ~0.333.
- CasADi-OSQP repeated-QP stateful/cold median ratios: n=20 ~0.978, n=60 ~0.945, n=120 ~0.847.

## Release boundary
M11 makes the project **benchmark-transfer-ready**, not MIPLIB/QPLIB-benchmarked. The next credible milestone is to take the 0.0.12 wheel to a machine with normal network access, install dedicated highspy/OSQP/PySCIPOpt, acquire/verify MIPLIB, and run the frozen harness. QPLIB execution requires a strict supported-subset parser/filter first.
