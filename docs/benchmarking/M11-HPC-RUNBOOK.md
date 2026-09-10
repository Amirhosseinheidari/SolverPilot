# M11 Portable Benchmark / HPC Runbook

This runbook is for a machine with normal network access and enough disk/CPU to run public optimization suites. The M11 harness is intentionally usable without Benchopt; Benchopt remains an optional higher-level orchestration layer.

## 1. Create a clean environment

```bash
python -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install './optimind_core_codename-0.0.12-py3-none-any.whl[open-source,benchmark]'
```

For the verification-only CasADi bridges as well:

```bash
pip install './optimind_core_codename-0.0.12-py3-none-any.whl[casadi,nlopt]'
```

## 2. Record machine and backend health

```bash
optimind-bench doctor --output run/doctor.json
```

Do not benchmark a native backend that is `unavailable` or `unhealthy`.

## 3. Acquire MIPLIB 2017 Benchmark v2

The built-in spec is pinned to:
- `benchmark.zip` (benchmark set version 2)
- `benchmark-v2.test`
- `miplib2017-v36.solu` (2026-01-26)

```bash
optimind-bench acquire \
  --dataset miplib2017-benchmark-v2 \
  --target datasets/miplib2017 \
  --retries 5

optimind-bench verify-dataset --target datasets/miplib2017
```

The acquisition lock records the observed SHA-256 of every downloaded artifact. Official MIPLIB currently does not publish a SHA-256 beside these download entries, so the lock is provenance/audit evidence rather than an upstream signed checksum.

## 4. Local benchmark run

Example direct portfolio plus deployable planner candidate:

```bash
optimind-bench run \
  --dataset-dir datasets/miplib2017 \
  --manifest datasets/miplib2017/benchmark-v2.test \
  --reference datasets/miplib2017/miplib2017-v36.solu \
  --backend highspy-native \
  --backend pyscipopt-native \
  --backend @auto \
  --repetitions 3 \
  --time-limit-s 3600 \
  --hard-timeout-s 3720 \
  --output run/miplib.jsonl
```

`time-limit-s` is the solver-level budget. `hard-timeout-s` is an independent controller-enforced process timeout and should be slightly larger to allow startup/serialization/cleanup.

## 5. Summarize

For deployable end-to-end behavior:

```bash
optimind-bench summarize \
  --input run/miplib.jsonl \
  --cutoff-s 3600 \
  --cost-field worker_solve_wall_s \
  --bootstrap-draws 10000 \
  --output run/miplib-summary.json
```

The summary contains:
- per-backend solved/validated/reference-match counts;
- PAR-style costs with explicit cutoff penalty;
- Single Best Solver (SBS);
- Virtual Best Solver (VBS);
- Dolan-Moré style performance-profile values;
- deployable `@auto` policy/SBS ratio and paired bootstrap 95% interval, when eligible.

`@auto` is never counted as a Single Best Solver candidate.

## 6. Research selector evaluation without touching production planner

Prepare a frozen policy JSON:

```json
{
  "50v-10": {"backend": "highspy-native", "overhead_s": 0.012},
  "drayage-25-23": {"backend": "pyscipopt-native", "overhead_s": 0.015}
}
```

It must cover every benchmark instance exactly once.

```bash
optimind-bench evaluate-policy \
  --input run/miplib.jsonl \
  --policy frozen-policy.json \
  --cutoff-s 3600 \
  --cost-field worker_solve_wall_s \
  --bootstrap-draws 10000 \
  --output run/policy-evaluation.json
```

Policy overhead is added to selected solver cost. This is the preferred path for holdout/ML research until a policy has earned production integration.

## 7. SLURM array

```bash
optimind-bench emit-slurm \
  --dataset-dir /shared/miplib2017 \
  --manifest /shared/miplib2017/benchmark-v2.test \
  --reference /shared/miplib2017/miplib2017-v36.solu \
  --backend highspy-native \
  --backend pyscipopt-native \
  --backend @auto \
  --output-dir /shared/results/optimind-m11 \
  --output run-miplib.slurm \
  --shards 32 \
  --repetitions 3 \
  --time-limit-s 3600 \
  --walltime 04:00:00 \
  --memory 16G
```

Submit with `sbatch run-miplib.slurm`.

Sharding is deterministic from instance filename. Changing shard count does not change protocol identity, so runs can be redistributed without changing experiment semantics.

## 8. Merge shards

```bash
optimind-bench summarize \
  --input results/shard-0.jsonl \
  --input results/shard-1.jsonl \
  --input results/shard-2.jsonl \
  --cutoff-s 3600 \
  --cost-field worker_solve_wall_s \
  --output results/summary.json
```

The merger rejects duplicate run IDs and refuses to summarize mixed protocol fingerprints.

## 9. Bundle audit artifacts

```bash
optimind-bench bundle \
  --file run/doctor.json \
  --file results/summary.json \
  --file results/shard-0.jsonl \
  --output optimind-miplib-run.zip \
  --label miplib-2026-run
```

The bundle contains its own SHA-256 manifest.

## QPLIB boundary in M11

`qplib-current` acquisition is implemented and verifies archive provenance/counts after extraction. **Execution is not yet implemented** because the M11 canonical QP IR supports a quadratic objective with linear constraints, whereas QPLIB also contains quadratic constraints and discrete QPs. Do not label an acquired QPLIB archive as benchmarked until a strict supported-subset parser/filter is implemented and executed.
