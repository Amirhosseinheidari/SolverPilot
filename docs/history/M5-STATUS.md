# M5 Status — MPS Ingestion, Manifest-Driven Benchmarking, and Selection Stability

**Date:** 2026-08-31  
**Version:** 0.0.6  
**Status:** Engineering/research milestone completed within the current package/network constraints.

## What M5 actually completed

### 1. Standard linear/MILP MPS ingestion into the canonical IR

M5 adds `parse_mps(text)` and `read_mps(path)` with transparent `.mps` / `.mps.gz` input.

Supported constructs:

- `NAME`;
- `OBJSENSE`;
- `OBJNAME`;
- `ROWS` (`N`, `E`, `L`, `G`);
- `COLUMNS` with duplicate-entry summation;
- integer `MARKER` / `INTORG` / `INTEND` cards;
- `RHS`;
- `RANGES` with row-type/sign semantics;
- `BOUNDS`: `LO`, `UP`, `FX`, `FR`, `MI`, `PL`, `BV`, `LI`, `UI`;
- `ENDATA`;
- Fortran `D` exponents.

Deliberately rejected rather than silently lowered:

- quadratic MPS sections (`QMATRIX`, `QUADOBJ`, `QSECTION`, `DMATRIX`);
- `SOS`;
- `INDICATORS`;
- `GENCONS`;
- semi-continuous/semi-integer bounds (`SC`, `SI`).

The parser preserves variable/constraint names in metadata and maps objective-row RHS values to the MPS objective offset convention (`offset = -RHS(objective_row)`).

### 2. Native semantic cross-check of the MPS reader

A randomized cross-check generated **200** bounded feasible MPS models covering LP and MILP semantics, including:

- minimization and maximization;
- integer markers;
- binary and general-integer bounds;
- ranged rows;
- objective offsets;
- scientific `E` and legacy `D` exponent syntax.

Each file was solved twice:

1. parse through OptiMind canonical IR, then solve with the development native HiGHS verifier;
2. read the same MPS file directly through the native HiGHS MPS reader.

Result:

```text
cases:                     200
passed:                    200
failed:                    0
max objective abs diff:    7.105427357601002e-15
```

Direct native HiGHS candidate solutions were also independently validated against the canonical IR.

This is strong format-semantics evidence for the supported MPS subset. It is not evidence for unsupported MPS extensions.

### 3. Manifest-driven public benchmark runner

M5 adds a benchmark runner designed around MIPLIB-style inputs:

- strict one-file-per-line `.test` manifest parsing;
- strict MIPLIB `.solu` parsing for `=opt=`, `=best=`, `=inf=`, `=unkn=`;
- direct `.mps.gz` parsing;
- explicit per-instance state: `solved`, `missing`, `parse_error`, `solve_error`;
- no silent instance dropping;
- separate parse/solve/end-to-end timing;
- canonical validation fields;
- optimal-reference objective checking;
- sense-aware “meets or beats best known” checking;
- infeasibility-reference checking;
- separate `match`, `mismatch`, and `inconclusive` accounting.

A deterministic seven-instance smoke suite (including gzip, LP, binary MILP, general integer, ranges, infeasible, max/best-known) produced:

```text
instances:              7
solved:                 7
reference checks:       7
reference matches:      7
reference mismatches:   0
```

### 4. M4 selection evidence stress-tested for timing stability

M5 did not promote the synthetic aspect-ratio rule into the production planner. Instead it re-measured the same structural holdout with interleaved solver order over **5 rounds per solver/instance**:

```text
24 instances
2 HiGHS algorithms
5 rounds each
240 direct solver calls
```

Results:

- median winner count: DS 12 / IPM 12;
- 23/24 instances had the same winner in at least 4 of 5 rounds;
- policy/SBS mean-cost ratio: **0.92195**;
- bootstrap 95% interval: **[0.87536, 0.97088]**;
- SBS→VBS gap closure: **0.8763**.

This confirms the M4 timing signal is not explained by a single lucky run. It remains synthetic, within-HiGHS algorithm-selection evidence and is still **not** activated in the production planner.

### 5. Reoptimization and prior property regressions re-run

After all M5 changes:

- mutation/hash property checks: **1000/1000** passed;
- OSQP translation property checks: **1200/1200** passed;
- native HiGHS repeated-LP benchmark: all expected reuse steps applied and all warm/cold objectives agreed.

Latest median stateful/cold end-to-end wall ratios:

- small: **0.6435**;
- medium: **0.4281**;
- large: **0.3361**.

These current measurements replace prior timing snapshots when describing the exact M5 execution.

## Current test state

Before release packaging, the full suite passes with only the three known optional public-native package skips:

1. `highspy` public-package integration;
2. `osqp` public-package integration;
3. `pyscipopt` public-package integration.

The skips are not counted as passes.

## Public benchmark status

MIPLIB 2017 benchmark-set **execution is still not completed** in this runtime.

The official benchmark v2 manifest contains 240 instances, and the current official solution-reference file is v36 dated 2026-01-26. The binary benchmark archive is about 317 MB. The current execution environment can access the official metadata/text pages but cannot retrieve the binary archive, so M5 does not claim a MIPLIB solve.

QPLIB execution is also still pending. Current official statistics list 134 continuous instances, 32 of them convex. Only the continuous-convex subset is in scope for the current QP core, and additional filtering is required because some convex continuous QPLIB instances include quadratic constraints that the current `QuadraticProblem` IR does not represent.

## What M5 does NOT prove

M5 does not prove:

- cross-solver selection advantage on MIPLIB/QPLIB/Netlib;
- that the synthetic M4/M5 selection rule belongs in production;
- exact public `highspy 1.15.1`, OSQP 1.1.3, or PySCIPOpt 6.2.1 runtime execution;
- support for every MPS extension used by every solver;
- QPLIB parser/import support;
- Benchopt execution;
- learned selection/configuration value.

## Gate decision

M5 removes a major engineering blocker for public MILP benchmarking: **MIPLIB-format data can now be read directly into the canonical IR and evaluated through a strict manifest/reference runner.**

The next hard gate is no longer benchmark plumbing. It is acquisition/execution of the actual public datasets and genuinely independent solver packages.
