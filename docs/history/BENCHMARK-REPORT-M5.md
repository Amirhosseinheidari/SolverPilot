# Benchmark Report — M5

**Date:** 2026-08-31  
**Version:** 0.0.6

## A. MPS semantic cross-check

Purpose: test whether the new canonical MPS reader agrees with a native HiGHS file reader, rather than merely passing hand-written parser tests.

Protocol:

- 200 deterministic random seeds;
- bounded feasible LP/MILP models;
- 4–12 variables and 2–8 rows;
- random `L/G/E` rows;
- optional `RANGES`;
- min/max objective;
- objective offsets through RHS objective row;
- binary/general integer declarations;
- E/D scientific notation;
- solve canonical model with stateful development HiGHS binding;
- independently read the same MPS file with native HiGHS;
- validate native HiGHS vector against canonical problem;
- compare objectives.

Result:

```text
200 / 200 passed
0 semantic discrepancies
max |objective difference| = 7.105427357601002e-15
```

Raw result: `benchmarks/results/m5-mps-parser-crosscheck.json`.

## B. Manifest/reference runner smoke

Purpose: verify benchmark accounting before touching MIPLIB.

Coverage:

- regular `.mps` and `.mps.gz`;
- LP min/max;
- objective offset;
- binary MILP;
- general integer;
- ranged constraint;
- infeasible reference;
- best-known reference.

Result:

```text
7 instances
7 solved
7 checkable references
7 matches
0 mismatches
0 inconclusive
```

The runner separately unit-tests missing and malformed files so they are recorded rather than dropped.

Raw result: `benchmarks/results/m5-manifest-runner-smoke.json`.

## C. Selection timing-stability stress test

Purpose: determine whether the positive M4 DS-vs-IPM holdout result was a one-run timing artifact.

Protocol:

- same 6 structural families and 24 fixed instances as M4;
- 5 interleaved timing rounds per algorithm/instance;
- algorithm order alternates by seed+round;
- median direct time per instance;
- leave-one-family-out rule selection;
- inspection cost included in policy cost;
- 10,000 bootstrap resamples of policy/SBS mean-cost ratio.

Result:

```text
solver calls:                                  240
median winners:                                DS 12 / IPM 12
instances with >=4/5 same round winner:        23 / 24
policy / SBS mean-cost ratio:                  0.9219546923
bootstrap 95% CI:                              [0.8753617636, 0.9708831333]
SBS -> VBS gap closure:                        0.8763039898
```

Interpretation:

- the structural complementarity is timing-stable on this controlled corpus;
- the deployable research policy still beats the synthetic SBS after paying inspection cost;
- this remains within-HiGHS algorithm selection on generated LPs;
- the production planner remains `capability_only` until public/OOD evidence exists.

Raw result: `benchmarks/results/m5-lp-selector-stability.json`.

## D. Reoptimization regression

Latest M3 benchmark rerun after M5 code changes:

| size | median stateful/cold wall ratio | approx speedup | median iteration ratio |
|---|---:|---:|---:|
| small | 0.6435 | 1.55x | 0.1450 |
| medium | 0.4281 | 2.34x | 0.1720 |
| large | 0.3361 | 2.98x | 0.1647 |

All warm/cold objective comparisons agree and every expected reuse step was reported as applied.

Raw result: `benchmarks/results/m3-highs-reoptimization.json`.

## E. Public corpus blocker

No MIPLIB or QPLIB performance result is reported in M5.

The benchmark runner is ready, but binary dataset retrieval remains blocked in the current runtime. Synthetic results are not relabeled as public-benchmark results.
