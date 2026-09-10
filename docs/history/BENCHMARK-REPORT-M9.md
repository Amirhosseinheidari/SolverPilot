# BENCHMARK REPORT — M9

## 1. Cross-library continuous LP

Backends:
- SciPy/HiGHS dual simplex (reference specialized LP solver);
- public NLopt 2.11.0 `LD_SLSQP`.

Random bounded feasible LPs include equality, lower-only, upper-only and ranged rows.

Results:
- 200/200 matched;
- max objective absolute difference: `3.144e-13`;
- max NLopt feasibility violation: `2.203e-13`.

This validates the new adapter's semantics. It is not evidence that NLopt should replace a dedicated LP solver.

## 2. Cross-library convex QP

48 QPs were generated from exact KKT constructions across four families (`interior`, `equality`, `upper_active`, `box_active`) and sizes 5/20/50.

- direct solves: 288
- portfolio executions: 48
- backend attempts inside portfolio: 96
- SciPy failures: 0
- NLopt failures: 0
- validated portfolio results: 48/48
- max objective error vs known KKT solution: `2.724e-11`
- max coordinate infinity-norm error: `2.473e-06`

Timing wins were SciPy 48 / NLopt 0 / ties 0. This result is deliberately not converted into a production selection rule.

## 3. Synthetic LP selector regression

- instances: 24
- solver calls: 240
- DS/IPM median winners: 13 / 11
- policy/SBS: `0.931`
- bootstrap 95%: `[0.890, 0.974]`
- SBS→VBS gap closure: `0.876`

Still synthetic and within HiGHS; not production evidence.

## 4. Reoptimization regression

Median stateful/cold wall ratios:
- small: `0.646` (~1.55x)
- medium: `0.430` (~2.33x)
- large: `0.342` (~2.93x)

All warm/cold objectives agreed and expected reuse steps were backend-confirmed.

## 5. Network/native solver gate

M9 retried official locked wheel URLs for highspy, OSQP and PySCIPOpt. Each failed at DNS resolution (`curl` return code 6). This is recorded in `benchmarks/results/m9-native-wheel-network-attempt.json`; none is counted as installed or verified.
