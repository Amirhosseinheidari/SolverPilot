# INTERNET / LITERATURE AUDIT M29

Verified during M29:

1. **MIPLIB 2017 feature methodology** — official selection methodology and the Mathematical Programming Computation paper. The paper explicitly discusses representation sensitivity and max-absolute normalization of objective/constraint rows before feature computation.
2. **ASlib** — algorithm selection evaluation includes feature computation cost and comparison against SBS/VBS; feature-cost files are first-class scenario data.
3. **SATzilla / SAT feature analysis** — static and probing features are distinct; probing may add task-relevant information but costs materially more.
4. **Run2Survive** — runtime selection data can be censored and risk-sensitive; avoiding catastrophic outcomes can be more important than mean prediction error.
5. **LP decision-tree algorithm selection work** — demonstrates that LP selection is possible in principle under some representations/datasets, preventing an overbroad conclusion from M29.
6. **2026 LP presolve study** — reports method-dependent effects of presolve on simplex/barrier performance, supporting presolve signals as a plausible hypothesis while not validating M29's probe empirically.

Claim boundary: literature motivates feature families and evaluation methodology. It does not override M29's measured negative cross-cohort result.
