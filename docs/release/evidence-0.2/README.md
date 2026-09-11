# 0.2 local validation evidence

Full installed-wheel regression: **1123 passed, 30 skipped**, Python 3.12.14, Windows.
An additional metadata-cache refresh regression passed after that full run (20 upgrade tests).
All 17 public examples ran successfully. Optional/external evidence skips remain explicit.
Coverage: 14,559 / 17,116 lines (85.1%); 4,299 / 6,188 branches (69.5%).

## Latency comparison

Single host: Intel i7-12650H, Windows 11, 15.63 GiB RAM. Synthetic workloads, seed 20260912;
two warmups and nine randomized-order measurements per method, one native thread.
Baseline 0.1.0rc2 package code equals published 0.1 except its version metadata.
Both use the same interpreter/dependencies. OS scheduling and thermal variation remain possible.

| Workload | Operation | 0.1 baseline median ms | 0.2 wheel median ms |
|---|---|---:|---:|
| LP-32 | direct | 1.493 | 1.485 |
| LP-32 | execute | 2.156 | 2.951 |
| LP-256 | direct | 7.794 | 8.089 |
| LP-256 | execute | 11.795 | 10.576 |
| LP-256 | auto | 32.746 | 14.674 |
| LP-256 | auto_shared_registry | 19.081 | 11.574 |
| LP-1500 | direct | 565.244 | 591.488 |
| LP-1500 | execute | 597.949 | 598.303 |
| QP-256 | construct | 3.920 | 2.620 |
| QP-256 | fresh | 2.287 | 2.291 |
| QP-256 | reuse | 1.806 | 1.610 |

Automatic LP-256 solving improved about 55%; QP construction about 33%.
Small LP-32 execution became about 37% slower with the added independent numerical KKT check.
LP-1500 total elapsed time was essentially unchanged; direct solver timing also varied.
These measurements do not establish universal speedups.

## Repeated compilation and memory

Three seeds, separate cached/full models, alternating order, identical resulting data hashes:
objective-only cached/full ratios 0.083–0.091; localized RHS 0.079–0.088; full matrix 0.944–0.970.
Full matrix changes therefore have no substantial demonstrated cache advantage.

For one reused QP backend, retaining only the latest result, RSS moved from 91.602 MiB
at solve 100 to 91.629 MiB at solve 1100. This is a bounded experiment, not a proof of no leaks.

## Numerical and application checks

New tests include 80 analytic LP cases, 100 changing-objective QPs, 30 PSD eigenvalue oracles,
cone boundary rejection, tampered certificate rejection, concurrent sessions, cancellation and cleanup.
Application examples exercise production allocation with demand penalties and ordered scenario batches.
These are analytic/synthetic application checks, not an industrial benchmark suite.

GitHub CI, exact-artifact qualification and publication are separate gates; this file records local evidence.
Raw samples and the compiler comparison script are included alongside this report.
