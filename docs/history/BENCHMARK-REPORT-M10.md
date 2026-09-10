# Benchmark Report M10

## Convex QP — four implementations
- Instances: 48
- Direct solver calls: 576
- Failures: `{"osqp": 0, "highs": 0, "scipy_slsqp": 0, "nlopt_slsqp": 0}`
- Max objective absolute error: `{"osqp": 2.220446049250313e-15, "highs": 2.708944180085382e-14, "scipy_slsqp": 8.763212377971286e-12, "nlopt_slsqp": 4.446221169018827e-11}`
- Timing wins: `{"osqp": 9, "highs": 3, "scipy_slsqp": 36, "nlopt_slsqp": 0, "ties": 0}`
- Portfolio validated: 48/48

The timing benchmark explicitly disables OSQP primal warm-start so solver timing is not contaminated by cross-instance reuse.

## Binary MILP — exact brute force + public p0033
- Synthetic instances: 100
- Direct solver calls: 909
- Failures: `{"scipy_highs": 0, "casadi_highs": 0, "cbc": 0}`
- Max objective error vs brute force: `{"scipy_highs": 3.552713678800501e-14, "casadi_highs": 3.552713678800501e-14, "cbc": 0.0}`
- Timing wins: `{"scipy_highs": 100, "casadi_highs": 0, "cbc": 0, "ties": 0}`
- p0033 reference objective: 3089
- p0033 objectives: `{"scipy_highs": 3089.0000000000077, "casadi_highs": 3089.0000000000077, "cbc": 3089.0}`

## Stateful OSQP bridge
The stateful benchmark reuses the same CasADi OSQP Function and explicitly passes the previous solution as `x0`; it does not claim internal factorization reuse.

`{"20": {"median_ratio": 0.9743897169201389, "median_speedup": 1.0262834086147896, "reuse_counts": [14, 14, 14, 14, 14], "max_objective_diff": 3.3306690738754696e-16}, "60": {"median_ratio": 0.9137609976724994, "median_speedup": 1.094378073202036, "reuse_counts": [14, 14, 14, 14, 14], "max_objective_diff": 8.881784197001252e-16}, "120": {"median_ratio": 0.8050950722484651, "median_speedup": 1.2420893313968566, "reuse_counts": [14, 14, 14, 14, 14], "max_objective_diff": 2.220446049250313e-15}}`

## Interpretation
Correctness diversity is now demonstrated across OSQP, HiGHS, CBC, SciPy-SLSQP and NLopt-SLSQP paths. Performance results remain environment/workload-specific and are not activated as a production planner policy.
