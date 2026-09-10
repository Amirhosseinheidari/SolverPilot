# Internet / Literature Audit — M24

## MIPLIB 2017

Official MIPLIB documentation describes the 240-instance Benchmark Set as a deliberately selected solver-performance benchmark, chosen for diversity, benchmark suitability, numerical stability, and coverage of feature/performance space.

Sources:
- https://miplib.zib.de/
- https://miplib.zib.de/Selection_Methodology.html
- Gleixner et al., *MIPLIB 2017: Data-Driven Compilation of the 6th Mixed-Integer Programming Library*, DOI 10.1007/s12532-020-00194-3.

M24 uses resource-bounded **LP relaxations derived from** this benchmark. It does not claim to reproduce the official MIPLIB MILP scoring protocol.

## Algorithm selection methodology

ASlib formalizes per-instance algorithm selection and explicitly records algorithm runs, features, cross-validation splits, and feature computation costs. ASlib warns that feature cost matters when comparing a selection policy to the best single algorithm.

Sources:
- Bischl et al., *ASlib: A benchmark library for algorithm selection*, Artificial Intelligence 237 (2016), DOI 10.1016/j.artint.2016.04.003.
- https://github.com/coseal/aslib-spec/blob/master/format.md

M24 therefore includes OptiMind inspection time in the deployed policy cost.

## HiGHS algorithms

Current HiGHS documentation exposes simplex and interior-point LP algorithms separately and notes that solver choice is problem-dependent. HiGHS' current solver documentation also documents simplex/IPM distinctions and crossover behavior.

Sources:
- https://ergo-code.github.io/HiGHS/dev/solvers/
- https://ergo-code.github.io/HiGHS/dev/options/definitions/

M24 compares the SciPy-exposed HiGHS dual-simplex and IPM paths only. It makes no cross-solver superiority claim.
