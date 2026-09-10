# Internet / Literature Audit — M26

M26 follows three external methodological constraints:

1. Algorithm-selection evaluation should account for feature computation cost rather than comparing only selected-solver runtime. ASlib explicitly incorporates feature costs into algorithm-selection scenario performance.
2. Recent work on feature-computation budget in per-instance algorithm selection shows that feature cost can materially explain loss relative to a virtual best solver, reinforcing explicit overhead accounting.
3. BenLOC highlights train/test leakage, homogeneous data, and inconsistent evaluation protocols as sources of over-optimistic learned-optimizer claims; M26 therefore freezes a group-safe split and stops before test when validation fails.

M26 does not use external literature to fill missing benchmark outcomes or alter the observed M25/M26 result.
