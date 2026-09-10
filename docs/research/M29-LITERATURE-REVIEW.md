# M29 LITERATURE REVIEW — Feature Value for Algorithm Selection

## Scope

This review informed the M29 feature families and claim boundaries. It does **not** turn the M29 post-outcome audit into held-out confirmation.

## 1. MIPLIB 2017: representation-aware mathematical-program features

Gleixner et al., *MIPLIB 2017: data-driven compilation of the 6th mixed-integer programming library*, Mathematical Programming Computation (2021), DOI 10.1007/s12532-020-00194-3.

Relevant methodological points:

- MIPLIB 2017 deliberately built a broad feature space to characterize instance structure.
- The paper explicitly warns that MPS representation/scaling can change raw features even for mathematically equivalent models.
- It normalizes the objective and each constraint row by the maximum absolute coefficient before feature computation.
- It includes matrix coefficients, sides, variable bounds, constraint structure and decomposition-related information; features are collected after trivial presolving.

M29 therefore treats representation invariance as a correctness property, not a modeling preference.

Official sources:
- https://miplib.zib.de/Selection_Methodology.html
- https://link.springer.com/article/10.1007/s12532-020-00194-3

## 2. ASlib: feature cost belongs in selector performance

Bischl et al., *ASlib: A Benchmark Library for Algorithm Selection* (2016; arXiv:1506.02465).

ASlib formalizes algorithm-selection scenarios with algorithm performance, instance features, cross-validation and optional feature-group costs. Its evaluation discussion explicitly adds feature computation time to selected-algorithm runtime and compares selectors against SBS/VBS. It also uses misclassification penalty relative to the per-instance best algorithm.

This directly motivates M29's cumulative feature-family cost accounting and the zero-feature-cost counterfactual diagnostic.

Source: https://arxiv.org/abs/1506.02465

## 3. SATzilla and probing/landmarking

Xu et al., *SATzilla: Portfolio-based Algorithm Selection for SAT*, JAIR 32 (2008).

SATzilla combines inexpensive structural features with probing features derived from bounded search behavior. This supports M29's decision to evaluate a solver-native presolve-only probe as a distinct family rather than conflating probing with static features.

Later feature analyses also show the trade-off: probing can add useful information but is substantially more expensive, and extra graph features can be redundant or encourage overfitting depending on the task.

Sources:
- https://www.jair.org/index.php/jair/article/view/10543
- https://ojs.aaai.org/index.php/SOCS/article/view/27292

## 4. Runtime uncertainty and risk

Tornede et al., *Run2Survive: A Decision-theoretic Approach to Algorithm Selection based on Survival Analysis*, ACML/PMLR 129 (2020).

Runtime observations can be censored/noisy, and risk-averse selection should avoid catastrophic timeout decisions rather than optimize only average prediction error. M29 therefore audits target repeatability and opportunity concentration instead of assuming a single timing label is truth.

Source: https://proceedings.mlr.press/v129/tornede20a.html

## 5. LP-specific evidence that algorithm selection can be possible

Vilas Boas et al., *Optimal Decision Trees for the Algorithm Selection Problem: Integer Programming Based Approaches* (2019), arXiv:1907.02211, studied algorithm selection for LP algorithm configurations over MIPLIB-related instances and reported cross-validated gains.

This is evidence that LP algorithm selection is not inherently impossible. It is **not** evidence that OptiMind's present feature representation is sufficient. M29 tests that question directly and finds poor cross-cohort portability.

Source: https://arxiv.org/abs/1907.02211

## 6. Presolve can affect LP methods differently

Recent work on Fourier-Motzkin based presolve reports that presolve transformations can affect primal simplex, dual simplex and barrier differently, supporting the hypothesis that presolve-reduction signatures may carry algorithm-relevant information. M29 therefore includes bounded presolve reduction features.

However, M29 finds that this probe is too expensive in the current implementation and does not transfer strongly enough to satisfy the authorization gate.

Source: https://link.springer.com/article/10.1007/s12532-026-00316-3

## M29 synthesis

The literature supports all four principles used here:

1. features must be representation-aware;
2. feature cost must be charged;
3. probing may help but must be budgeted separately;
4. runtime uncertainty and cross-instance generalization matter more than in-sample correlation.

M29's negative conclusion is therefore not "ML can never select LP algorithms". It is narrower:

> Under the current OptiMind representation, current implementation cost, and two non-overlapping public/OOD cohorts, the richer feature families do not provide sufficiently stable, transferable value to justify another selector experiment for version 1.0.
