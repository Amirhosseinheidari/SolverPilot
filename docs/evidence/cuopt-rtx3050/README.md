# RTX 3050 GPU LP qualification

GPU execution passed on the tested LP cases. **A speed advantage over the best
tested CPU alternative did not pass:** CPU OR-Tools PDLP was faster than the new
isolated cuOpt path at every tested size. Automatic GPU selection stays disabled.

## Machine, software and scope

- Windows 11 host, WSL 3.0.1 / Ubuntu 24.04, Python 3.12.
- NVIDIA GeForce RTX 3050 Laptop GPU, 4 GiB VRAM, Windows driver 566.07.
- Intel Core i7-12650H; the Linux guest has about 7.6 GiB RAM. Each solver was
  configured for one CPU thread, and the GPU worker used explicit FP64 PDLP.
- cuOpt 26.8.0 / CUDA runtime 12.9; HiGHS 1.15.1; OR-Tools 9.15.6755.
- Continuous bounded sparse LP only. No QP/MIP, warm-start, native context reuse,
  general driver compatibility or industrial scalability is qualified here.

This is an unreleased source extension after 0.4. The local qualification wheel
still has the development metadata version 0.4; it is **not** the published PyPI
0.4 wheel. Its hash and native test results are in [validation.json](validation.json).
The CPU-PDLP comparison imported that installed wheel from the Linux environment,
outside the checkout. The adapter and worker hashes agree across all campaigns.

## Measured results

Each campaign used three deterministic seeds at each size and alternated GPU/CPU
execution order. All 27 recorded GPU solves passed primal validation and the
independent original-domain numerical optimality check. All 9 CPU-PDLP solves
passed the same checks. The models have half as many rows as variables and four
nonzeros per variable on average, with a planted primal/dual optimum.

The closest algorithm-family comparison is the installed-wheel PDLP campaign.
These are medians of complete `solve()` wall times in seconds, including fresh
worker startup and independent verification:

| Variables | Rows | Nonzeros | cuOpt GPU PDLP | CPU OR-Tools PDLP |
|---:|---:|---:|---:|---:|
| 500 | 250 | 2,000 | 2.710 | 0.232 |
| 5,000 | 2,500 | 20,000 | 2.610 | 0.484 |
| 20,000 | 10,000 | 80,000 | 3.169 | 1.357 |

GPU startup/import overhead is material. Native solver time alone must not be
compared with another backend's complete call. Even the two PDLP implementations
have different scaling, stopping and implementation choices, so this is a
backend comparison, not an isolated measurement of GPU hardware acceleration.

The additional HiGHS comparisons show why one CPU baseline is insufficient:

| Variables | HiGHS default verified | HiGHS IPM verified | Observation |
|---:|---:|---:|---|
| 500 | 3/3 | 3/3 | CPU medians about 0.049 / 0.039 seconds |
| 5,000 | 2/3 | 3/3 | Default had one timeout without a solution; IPM median 12.481 seconds |
| 20,000 | 0/3 | 0/3 | Both hit their configured 30-second native budgets without a solution |

One default-HiGHS 5,000-variable run reported a limit while its returned primal
and dual independently verified optimality. Time-limited runs without validated
solutions are retained as failures; their durations are not completed-solve
speedups. The comparison command intentionally exits nonzero when any baseline
fails its checks. This does not erase the separately successful GPU records.

## Reproduce and inspect

From the checkout containing this record and its qualification tool, follow
[the GPU installation guide](../../GPU-LP.md), then run:

```sh
SOLVERPILOT_TEST_CUOPT=1 python -m pytest tests/test_cuopt_native.py -ra
python tools/qualify_cuopt.py --output default.json
python tools/qualify_cuopt.py --cpu-solver ipm --output ipm.json
python tools/qualify_cuopt.py --cpu-solver pdlp --output pdlp.json
```

- [Default HiGHS campaign](gpu-qualification.json)
- [HiGHS IPM campaign](gpu-qualification-ipm.json)
- [CPU-PDLP campaign, installed development wheel](gpu-qualification-pdlp.json)
- [Complete package versions](packages.json)

The host recording date is 2026-09-29. The Linux guest wall clock was ahead of
the Windows host; its original `captured_at` values remain in the records with
an explicit note. Solve durations use monotonic `perf_counter` differences.
Scope labels and summary counts were normalized from the recorded backend names;
per-solve timings, model hashes and outcomes are retained.

These three-seed synthetic cases establish bounded functional conformance and
local observations. They are neither a held-out production routing dataset nor
evidence that every LP, NVIDIA driver or GPU will behave identically. The next
performance work should measure a persistent GPU context and representative
workloads against CPU PDLP as well as simplex/IPM before enabling any routing rule.
