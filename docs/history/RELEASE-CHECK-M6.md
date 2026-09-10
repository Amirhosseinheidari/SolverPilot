# Release Check — M6

**Date:** 2026-08-31  
**Version:** 0.0.7

The checks below were executed against the final M6 source tree before archive creation.

## Packaging

```text
editable install: PASS
installed version: 0.0.7
compileall src/tests/benchmarks/tools: PASS
wheel build: PASS
clean wheel install to isolated target: PASS
wheel-path infeasibility diagnostic smoke: PASS
```

Built wheel:

```text
dist/optimind_core_codename-0.0.7-py3-none-any.whl
SHA-256: dff1ad9b5fc9a8897a2877b5af92d7fa832af6874e3f52f203fd8391273705f5
```

The isolated-wheel smoke solved a contradictory LP with `diagnose_infeasible=True`, returned public status `infeasible`, attached a confirmed `InfeasibilityReport`, and produced a positive elastic-relaxation objective.

## Unit / integration contracts

```text
130 collected
127 passed
3 skipped
0 failed
```

Known skips only:

1. exact public `highspy` integration;
2. exact public `osqp` integration;
3. exact public `pyscipopt` integration.

## Property / semantic verification

```text
M1 mutation/hash checks:             1000 / 1000
M2 OSQP translation checks:          1200 / 1200
M5 MPS native cross-check:             200 / 200
M6 diagnostic property checks:         200 / 200
M6 native HiGHS IIS stress:              50 / 50
M5 manifest/reference smoke:               7 / 7
```

## Real external instance

Netlib AFIRO:

```text
reference objective:             -464.75314286
DS validated objective:          -464.7531428571429
IPM validated objective:         -464.7531428571429
abs error vs published ref:      2.8571e-09
native-file vs canonical diff:   5.6843e-14
```

The AFIRO file was used as a temporary external input and is not redistributed in the archive.

## Selector regression

```text
24 instances
240 direct solver calls
DS/IPM median winners: 13 / 11
23/24 instances have >=4/5 same timing-round winner
policy/SBS ratio: 0.9245710917445353
bootstrap 95%: [0.8803468793787841, 0.9703460837719281]
gap closure: 0.8788414088449568
```

Still synthetic within-HiGHS evidence; production planner remains `capability_only`.

## Reoptimization regression

```text
small stateful/cold wall ratio:   0.6034425732723145
medium stateful/cold wall ratio:  0.4120124491782514
large stateful/cold wall ratio:   0.3334235125054542
all expected reuse steps applied: yes
all warm/cold objectives agree: yes
```

## Explicitly not release-cleared

The following are **not** claimed complete:

- full Netlib benchmark execution;
- MIPLIB benchmark execution;
- QPLIB benchmark execution;
- exact public native solver-package execution for highspy/OSQP/PySCIPOpt;
- Benchopt execution;
- empirical cross-solver production selection policy;
- learned selection/configuration;
- public package naming/legal clearance.
