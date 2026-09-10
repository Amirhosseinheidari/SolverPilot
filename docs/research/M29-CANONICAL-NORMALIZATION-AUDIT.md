# M29 CANONICAL NORMALIZATION AUDIT

## Why this audit exists

During M29, representation sensitivity was treated as a potential scientific bug. Equivalent LPs can be written with arbitrarily rescaled rows or objective coefficients, and raw coefficient summaries can therefore create spurious feature signal.

## Literature contract

MIPLIB 2017 explicitly normalizes:

- every constraint row by its maximum absolute matrix coefficient, and applies the same scale to finite row sides;
- the objective coefficient vector by its maximum absolute coefficient.

It also uses signed-log scaling for signed bound-style features.

M29 adopts this representation principle for its canonical static feature family.

## Implementation checks

`benchmarks/m29_feature_worker.py` implements row/objective normalization before coefficient, RHS and norm summaries.

Two checks are release evidence:

1. deterministic unit test with an equivalent row/objective-scaled LP;
2. randomized 200-case property campaign with independent positive row scales and objective scale spanning approximately 16 orders of magnitude.

Final property result:

- 200 / 200 passed;
- max relative feature discrepancy < 1e-15 scale (floating-point noise only).

Raw result: `benchmarks/results/m29/m29-scaling-invariance-property.json`.

## Boundary

This does not make features invariant to arbitrary **column/variable transformations**. MIPLIB itself notes continuous-column scaling as another representational freedom. M29 does not claim complete formulation invariance; instead it removes the row/objective scaling artifacts that directly affect its chosen coefficient/side summaries.
