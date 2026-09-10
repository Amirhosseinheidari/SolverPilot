# Benchmark / Verification Report — M30

M30 is a consolidation milestone, not a comparative-performance milestone. It therefore reruns correctness and reuse campaigns but does not manufacture a new solver-superiority claim.

## Direct C-API correctness

### MILP / bundled HiGHS

- 100 exact small MILP instances
- 300 solver calls across direct HiGHS C API, SciPy/HiGHS, and CBC reference route
- failures: 0 for all three routes
- maximum objective absolute error for direct HiGHS: approximately `1.28e-13`

### Convex QP / bundled OSQP

- 96 constructed-KKT QPs
- 192 solver calls
- direct OSQP failures: 0
- CasADi/OSQP bridge failures: 0
- maximum direct objective absolute error: approximately `1.78e-15`
- maximum direct-vs-bridge objective difference: 0

## Stateful reoptimization rerun

Median stateful/cold ratios in the current controlled workload were:

LP:
- small: ~0.967
- medium: ~0.890
- large: ~0.779

QP:
- n=20: ~0.977
- n=60: ~0.911
- n=120: ~0.905

All objective comparisons agreed and all expected reuse events were observed. These timing values are environment/workload-specific and are **not** promoted to a general performance claim by M30.

## Research-branch decision carried into M30

M25–M29 established that a DS/IPM virtual-best opportunity exists on the investigated public/OOD cohorts, but the tested learned/selective feature representations did not generalize sufficiently. M30 therefore freezes learned LP performance auto-routing off for the 1.0 release line.
