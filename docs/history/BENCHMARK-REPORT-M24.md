# Benchmark Report — M24 Public/OOD Comparative LP Selection

## Question

Does the LP structural rule discovered on synthetic M4/M5 families generalize to public/OOD optimization structure strongly enough to justify production performance routing?

## Pre-registered policy

The rule was frozen before M24:

`choose scipy-highs-ipm if m/n >= 1.5, else scipy-highs-ds`

M24 did not tune the threshold on the public corpus.

## Public/OOD corpus construction

Source: official MIPLIB 2017 benchmark-v2 archive, integrity-verified during M22.

M24 evaluates **continuous LP relaxations derived from official MIPLIB MILP models**. This must not be described as a MIPLIB MILP solver comparison.

The resource envelope is outcome-independent and fixed before solving:

- archive entry compressed size <= 256 KiB,
- OptiMind canonical parser support required,
- n <= 2000,
- m <= 2000,
- nnz <= 80,000,
- deterministic SHA-256(filename) order,
- first 24 eligible instances,
- 3 rounds per algorithm/instance,
- DS/IPM order alternated,
- 0.75 s per-solve cutoff,
- median timing cost,
- independent validation through the normal `execute` path,
- inspector median cost added to policy cost.

One selected candidate was excluded by parser scope and 45 candidates by the predeclared structural/resource envelope before timing.

## Results

| Metric | Run 1 | Run 2 |
|---|---:|---:|
| Instances | 24 | 24 |
| DS wins | 22 | 22 |
| IPM wins | 1 | 1 |
| Ties | 1 | 1 |
| Objective mismatches | 0 | 0 |
| DS terminal rate | 95.83% | 95.83% |
| IPM terminal rate | 95.83% | 95.83% |
| SBS | DS | DS |
| SBS mean cost | 0.052515 s | 0.052465 s |
| VBS mean cost | 0.050970 s | 0.050980 s |
| Rule policy mean incl. inspection | 0.053539 s | 0.053496 s |
| Policy/SBS point ratio | 1.019510 | 1.019654 |
| Bootstrap 95% upper | 1.077470 | 1.078408 |
| SBS→VBS gap closure | -0.663 | -0.694 |
| Selector promoted | No | No |

The same cohort and same per-instance winner pattern occurred in both runs.

The rule selected IPM for `graphdraw-domain`, `supportcase26`, and `neos-3754480-nidda`. The only IPM median winner was `csched007`, so the public/OOD result is not merely a small threshold miss around the old rule boundary.

## Promotion gates

Passed in both runs:

- >=24 instances,
- DS terminal rate >=90%,
- IPM terminal rate >=90%,
- zero objective mismatches.

Failed in both runs:

- policy beats SBS point estimate,
- bootstrap 95% upper bound below 1,
- >=50% SBS→VBS gap closure.

Therefore **no performance selector is promoted**.

## Interpretation

M4/M5 remains valid evidence that DS/IPM complementarity exists on those controlled synthetic families. M24 demonstrates that the specific synthetic rule does not generalize to this resource-bounded public/OOD MIPLIB-derived LP cohort.

This is useful falsification evidence, not a project failure. The correct production action is to retain dual simplex as the conservative LP baseline and require stronger, genuinely generalizing evidence before enabling per-instance performance routing.
