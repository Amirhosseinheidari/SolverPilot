# Benchmark Report M8

## 1. Runtime resilience / trust-boundary benchmark

M8 adds a 100-case randomized LP fault-injection benchmark. A deliberately poisoned backend truthfully advertises LP capability but returns a mathematically invalid solution. A second injected backend raises an exception. Real SciPy HiGHS dual-simplex is used as the healthy recovery backend.

Results:

- capability-only invalid outcomes: **100/100**;
- health-aware validated outcomes: **100/100**;
- portfolio exception recovery: **100/100**;
- invalid portfolio selections: **0**.

This benchmark proves health-gating and failure isolation behavior. It is not intended to compare solver speed.

Raw: `benchmarks/results/m8-runtime-resilience.json`.

## 2. Solver-selection stability regression

The prior 24-instance synthetic structured-LP benchmark was rerun from M8 source:

- solver calls: **240**;
- DS median wins: **12**;
- IPM median wins: **12**;
- stable winner >=4/5 rounds: **22/24**;
- policy/SBS ratio: **0.923483**;
- bootstrap 95% interval: **[0.876471, 0.971705]**;
- SBS→VBS gap closure: **0.865801**;
- max objective difference: **0.0**.

Still research-only: this is algorithm selection inside HiGHS on synthetic LPs.

## 3. Native HiGHS repeated-LP reuse regression

Median stateful/cold wall ratios:

| Size | Ratio | Approx. speedup | Objective agreement | Expected reuse |
|---|---:|---:|---|---|
| small | 0.671828 | 1.49x | yes | all |
| medium | 0.424940 | 2.35x | yes | all |
| large | 0.344548 | 2.90x | yes | all |

The verifier still uses SciPy's private vendored HiGHS binding and remains development-only.

## 4. Correctness regressions

- mutation/hash: 1000/1000;
- OSQP translation: 1200/1200;
- MPS parser/native-reader randomized cross-check: 200/200, max objective difference `7.105427357601002e-15`;
- diagnostics property checks: 200/200;
- native IIS stress: 50/50.

## 5. Public-native integration blocker

For every package in `benchmarks/native-wheel-lock.json`, M8 attempted network reachability to its exact locked official wheel URL. All attempts returned DNS failure before any binary was downloaded. Details are preserved in `benchmarks/results/m8-native-wheel-network-attempt.json`.

Therefore there is still no honest cross-solver public-package benchmark in M8.
