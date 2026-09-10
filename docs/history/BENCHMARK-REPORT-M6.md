# Benchmark Report — M6

**Date:** 2026-08-31  
**Version:** 0.0.7

## A. Infeasibility diagnostic property checks

Protocol:

- 100 deterministic planted row-conflict LPs;
- each has a true contradiction `x_j >= L` and `x_j <= U` with `L > U`;
- 0–4 irrelevant wide constraints are added;
- require positive elastic optimum;
- require complete irreducible deletion-filter conflict of exactly the two planted row-bound sides;
- 100 deterministic integer variables with bounds strictly between consecutive integers;
- require static proof plus a complete two-bound deletion conflict.

Result:

```text
200 / 200 passed
max deletion-filter checks: 11
```

Raw result: `benchmarks/results/m6-diagnostics-property-checks.json`.

## B. Native IIS stress check

The SciPy-vendored native HiGHS development verifier was run on 50 separately generated infeasible LPs with a planted two-row conflict.

```text
valid IIS results: 50 / 50
failures:           0
```

This verifies the normalization plumbing against a real native HiGHS binary, but it is still the private SciPy-vendored HiGHS path, not public `highspy 1.15.1`.

## C. Netlib AFIRO real-instance smoke

Source MPS: `https://raw.githubusercontent.com/coin-or-tools/Data-Sample/master/afiro.mps`  
Reference table: `https://coap.math.ufl.edu/test-problems/`

The third-party instance was used only as a temporary local benchmark input and is not included in the distributed archive.

Input SHA-256 used for the run:

```text
3f9d8c753987b6ebfe87834a18b771012c0a6348e265cb31858f321b57ebce48
```

Reference optimum: `-464.75314286`.

| backend | validated objective | abs error vs reference | median end-to-end wall |
|---|---:|---:|---:|
| HiGHS dual simplex bridge | -464.753142857143 | 2.857e-09 | 2.866 ms |
| HiGHS IPM bridge | -464.753142857143 | 2.857e-09 | 2.976 ms |

A separate direct-file-reader cross-check on the identical MPS bytes gave:

```text
native HiGHS objective:   -464.75314285714285
canonical IR objective:   -464.7531428571429
absolute difference:      5.684e-14
canonical validation:     True
```

Raw results:

- `benchmarks/results/m6-netlib-afiro.json`
- `benchmarks/results/m6-netlib-afiro-native-crosscheck.json`

Interpretation: parser/runtime semantics work on a real historical Netlib LP. AFIRO is tiny and cannot support a solver-selection performance claim.

## D. MPS semantic regression

M5 randomized parser/native-reader cross-check rerun:

```text
200 / 200 passed
max objective absolute difference = 7.105427357601002e-15
```

## E. Selector stability regression

```text
solver calls:          240
DS / IPM wins:         13 / 11
stable instances:      23 / 24
policy/SBS ratio:      0.924571
bootstrap 95% CI:      [0.880347, 0.970346]
gap closure:           0.878841
```

This remains synthetic within-HiGHS research evidence only.

## F. Reoptimization regression

| size | stateful/cold median wall ratio | approximate speedup | median iteration ratio |
|---|---:|---:|---:|
| small | 0.6034 | 1.66x | 0.1450 |
| medium | 0.4120 | 2.43x | 0.1720 |
| large | 0.3334 | 3.00x | 0.1647 |

All expected reuse events were actually reported and all warm/cold objective comparisons agreed.

## G. Public-corpus status

M6 still does not report full Netlib, MIPLIB, or QPLIB aggregate performance. One real AFIRO run is intentionally reported as a smoke only.
