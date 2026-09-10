# M11 Benchmark / Harness Report

## Purpose
M11 does not claim a new solver-speed record. It validates the experiment infrastructure needed to make future public-corpus claims auditable and transferable.

## Portable harness smoke
Raw: `benchmarks/results/m11-harness-smoke.json`

Configuration:
- 18 deterministic generated LPs;
- direct candidates: SciPy HiGHS dual simplex, SciPy HiGHS IPM, CasADi HiGHS verification bridge;
- `@auto` candidate over the same direct portfolio;
- 2 repetitions;
- 3 deterministic shards;
- 144 expected result rows.

Observed direct metrics in the final M11 smoke run:
- SBS: `scipy-highs-ds`.
- SBS mean wall cost: 0.0031742183611337773 s.
- VBS mean wall cost: 0.0028895109167024202 s.

Deployable `@auto`:
- selected dual simplex on all 18 instances;
- mean policy cost: 0.0063084006111024694 s;
- policy/SBS ratio: 1.987387096094175;
- paired bootstrap 95% interval: [1.8085865428884182, 2.2566669898100518];
- gap closure: -11.008430974569556.

Interpretation: on these short instances, planner overhead dominates any selection benefit. This is negative evidence and is intentionally not converted into a production performance rule.

## Resume/shard validation
The three shards collectively covered all 18 instances without overlap. Re-running shard 0 under the same protocol skipped all 48 already-complete rows. Protocol fingerprints intentionally ignore shard count/index but include solver versions, dataset manifest/reference hashes, tolerances, budgets and seed; per-run identity additionally includes instance SHA-256.

## Hard-timeout isolation
Raw: `benchmarks/results/m11-isolation-smoke.json`

- normal subprocess-isolated solve: `solved`.
- forced controller timeout (`1e-6` seconds): `hard_timeout`.

This is separate from solver-level `time_limit_s` and protects a distributed benchmark from a hung/crashed binding.

## External policy evaluator smoke
Raw: `benchmarks/results/m11-policy-evaluator-smoke.json`

- fixed SBS policy with zero overhead reconstructed policy/SBS ratio 1.0.
- oracle-per-instance choices reconstructed gap closure 1.0.
- the oracle is explicitly leakage-only and is used only to verify evaluator arithmetic.

The evaluator requires exact instance coverage and adds per-instance selector/feature overhead to selected solver cost.

## MIPLIB acquisition attempt
Raw: `benchmarks/results/m11-miplib-metadata-acquisition-attempt.json`

A safe metadata-only acquisition attempted to fetch `benchmark-v2.test` and `miplib2017-v36.solu` without downloading the 317.3 MB archive. It failed with an external DNS/name-resolution `URLError`. Therefore no MIPLIB run is claimed in M11.

## Regression performance evidence retained
### Research DS/IPM selection stability
Fresh M11 run of the M5 research benchmark:
- instances: 24;
- solver calls: 240;
- median winners: DS 13, IPM 11;
- stable instances: 23/24;
- policy/SBS ratio point: ~0.9215;
- paired bootstrap interval: ~[0.8757, 0.9703];
- gap closure: ~0.8674.

This remains synthetic research evidence and is not the production planner.

### HiGHS repeated LP
Fresh M11 regression:
- small stateful/cold median wall ratio: ~0.6374;
- medium: ~0.4165;
- large: ~0.3326;
- all objectives agree;
- every expected reuse step was applied.

### CasADi-OSQP repeated QP
Fresh M11 regression:
- n=20 ratio ~0.9776;
- n=60 ratio ~0.9446;
- n=120 ratio ~0.8470;
- all expected reuse counts were 14/14;
- max warm/cold objective difference <=2.22e-15.

### Cross-solver QP
Fresh M11 regression of M10:
- 48 QPs;
- 576 direct calls;
- zero failures for OSQP, HiGHS-QP, SciPy SLSQP and NLopt SLSQP.

### Cross-solver exact binary MILP
Fresh M11 synthetic regression:
- 100 instances with brute-force optimum;
- zero failures across SciPy/HiGHS, CasADi/HiGHS and CBC;
- max objective error: 3.553e-14 for HiGHS paths, 0 for CBC.

## Claim boundary
The M11 harness is ready to carry MIPLIB results from a proper network/HPC machine. It does not transform missing public-corpus execution into evidence. QPLIB remains acquisition-only because a strict supported-subset QPLIB parser/filter is not yet implemented.
