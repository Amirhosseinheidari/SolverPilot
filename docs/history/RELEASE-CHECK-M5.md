# Release Check — M5

**Date:** 2026-08-31  
**Version:** 0.0.6

The checks below were executed against the final M5 source tree before archive creation.

## Packaging

```text
editable install: PASS
installed version: 0.0.6
public read_mps/parse_mps exports: PASS
compileall src/tests/benchmarks/tools: PASS
wheel build: PASS
clean wheel install to isolated target: PASS
wheel-path smoke solve from parsed MPS: PASS (valid_optimal, objective 2.0)
```

Built wheel:

```text
dist/optimind_core_codename-0.0.6-py3-none-any.whl
SHA-256: e297bc3e9b72c7a0680f736d0a768a5241dc7189431572ecef3a1515d0edb58b
```

## Unit / integration contracts

```text
116 passed
3 skipped
```

Known skips only:

1. exact public `highspy` integration;
2. exact public `osqp` integration;
3. exact public `pyscipopt` integration.

No failed tests were suppressed.

## Property / semantic verification

```text
M1 mutation/hash checks:       1000 / 1000
M2 OSQP translation checks:    1200 / 1200
M5 MPS native cross-check:      200 / 200
M5 manifest runner smoke:         7 / 7 references matched
```

## Performance/regression verification

M5 selector stability:

```text
24 instances
240 direct solver calls
DS/IPM median winners: 12 / 12
23/24 instances have >=4/5 same timing-round winner
policy/SBS ratio: 0.9219546922521669
bootstrap 95%: [0.8753617636331027, 0.9708831333354894]
gap closure: 0.8763039897828029
```

M3 reoptimization rerun after M5:

```text
small stateful/cold wall ratio:   0.6434628974588261
medium stateful/cold wall ratio:  0.42806461046114164
large stateful/cold wall ratio:   0.33606537719725854
all expected reuse steps applied: yes
all warm/cold objectives agree: yes
```

## Explicitly not release-cleared

The following are **not** claimed complete:

- MIPLIB benchmark execution;
- QPLIB benchmark execution;
- public native solver-package execution for highspy/OSQP/PySCIPOpt;
- empirical production solver-selection policy;
- public package naming/legal clearance.
