# M12 Benchmark Report

## 1. Benchmark-trust smoke
Synthetic short LP smoke only; this is **not** solver-performance evidence.
- rows: 48/48;
- mixed-environment timing aggregation rejected: `True`;
- fixed SBS external policy reconstructs policy/SBS ratio: `1.0`;
- deployable `@auto` point policy/SBS: `1.066099`;
- bootstrap 95%: `[0.908744, 1.243073]`.

The interval crosses 1, therefore this smoke supports no selector-performance claim.

## 2. Cross-solver convex QP regression
- 48 instances, 576 direct calls;
- failures: {'osqp': 0, 'highs': 0, 'scipy_slsqp': 0, 'nlopt_slsqp': 0};
- max objective abs error: {'osqp': 2.220446049250313e-15, 'highs': 2.708944180085382e-14, 'scipy_slsqp': 8.763212377971286e-12, 'nlopt_slsqp': 4.446221169018827e-11};
- timing wins: {'osqp': 7, 'highs': 6, 'scipy_slsqp': 32, 'nlopt_slsqp': 3, 'ties': 0};
- portfolio validated: 48/48.

Timing winners differ from historical M10 runs, reinforcing that millisecond-scale timing should not be promoted directly into production selection rules.

## 3. Exact binary MILP regression
- 100 synthetic binary MILPs with brute-force optima;
- 900 solver calls;
- failures: {'scipy_highs': 0, 'casadi_highs': 0, 'cbc': 0};
- max objective error vs brute force: {'scipy_highs': 3.552713678800501e-14, 'casadi_highs': 3.552713678800501e-14, 'cbc': 0.0};
- `p0033` fresh M12 state: `not_executed_missing_third_party_instance`. The third-party file is intentionally not redistributed, so no fresh p0033 claim is made in M12.

## 4. Stateful QP regression
- n=20: stateful/cold median ratio `1.038735`, speedup `0.963x`, reuse counts `[14, 14, 14, 14, 14]`, max objective diff `3.331e-16`.
- n=60: stateful/cold median ratio `0.898432`, speedup `1.113x`, reuse counts `[14, 14, 14, 14, 14]`, max objective diff `8.882e-16`.
- n=120: stateful/cold median ratio `0.918897`, speedup `1.088x`, reuse counts `[14, 14, 14, 14, 14]`, max objective diff `2.220e-15`.

This bridge evidence is explicitly `CasADi 3.7.2 + OSQP bridge` primal warm-start evidence; it is not a claim of factorization reuse.

## 5. HiGHS stateful LP regression
- small: wall ratio `0.616088`, bootstrap 95% `[0.5007116273115049, 0.6858084507271694]`, objective agreement `True`, all expected reuse `True`.
- medium: wall ratio `0.450802`, bootstrap 95% `[0.40966122880368433, 0.5137031904484693]`, objective agreement `True`, all expected reuse `True`.
- large: wall ratio `0.346447`, bootstrap 95% `[0.3413681583291403, 0.44686093670195093]`, objective agreement `True`, all expected reuse `True`.

## 6. Selector rerun boundary
The fresh M12 attempt of `m5_lp_selector_stability.py` exceeded the 180-second execution window. No partial output is used. Historical M5/M11 selector figures remain historical evidence only.

## 7. MIPLIB boundary
A fresh acquisition attempt through the installed 0.0.13 wheel failed with `URLError: Temporary failure in name resolution`. Therefore M12 contains **no MIPLIB performance result**.
