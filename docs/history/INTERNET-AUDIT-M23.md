# Internet / Research Audit — M23

M23 follows the classical algorithm-selection framing while deliberately requiring stronger evidence before automated performance routing.

## Foundational algorithm selection

John R. Rice, *The Algorithm Selection Problem*, Advances in Computers 15 (1976), 65–118. DOI: 10.1016/S0065-2458(08)60520-3.

## Per-instance portfolios

Xu, Hutter, Hoos, Leyton-Brown, *SATzilla: Portfolio-based Algorithm Selection for SAT*, JAIR 32 (2008), 565–606. DOI: 10.1613/JAIR.2490.

## Representative benchmark construction

MIPLIB 2017 uses feature/performance-aware data-driven selection to construct a diverse benchmark subset of 240 instances from a larger collection. Gleixner et al., Mathematical Programming Computation 13 (2021). DOI: 10.1007/s12532-020-00194-3.

## Modern MIP configuration/selection caution

BenLOC (2025) highlights data leakage, homogeneous datasets, and inconsistent evaluation as causes of over-optimistic MIP configuration claims: https://arxiv.org/abs/2506.02752

## Feature-cost accounting

van der Blom & Vermetten (2026) study feature-computation budget in per-instance algorithm selection and show that the feature budget can materially explain loss to the virtual-best solver: https://arxiv.org/abs/2605.04954

## M23 consequence

OptiMind does not convert M22 correctness/corpus evidence into a performance selector. Comparative held-out evidence, feature-cost accounting, fixed-environment provenance, and at least two candidate backends are required before automatic performance overrides are accepted.
