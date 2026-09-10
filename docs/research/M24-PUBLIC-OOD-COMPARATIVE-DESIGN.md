# M24 Public/OOD Comparative Design

## Motivation

M4/M5 found a stable synthetic DS-vs-IPM structural rule. M23 correctly refused to productionize it because no public/OOD comparative evidence existed. M24 is the falsification test.

## Anti-overfitting rules

- the selector threshold is frozen before public/OOD timing;
- public instances are never used to refit the threshold;
- cohort selection is based on archive size, parser/IR scope, structural resource limits, and a deterministic filename hash—not solver timing;
- inspection cost is charged to the policy;
- failures/time limits receive the declared cutoff cost rather than disappearing;
- promotion requires a bootstrap upper bound below the SBS baseline plus correctness/coverage gates;
- the experiment is repeated unchanged.

## Evidence-contract hardening

M24 revealed a semantic gap in M23: `COMPARATIVE_HELDOUT` described how data was collected, but not whether the resulting selector actually passed promotion criteria. M24 separates these concepts.

`ProductionEvidence.supports_performance_ranking` now requires all of:

1. comparative held-out evidence;
2. >=2 compared backends;
3. feature-cost accounting;
4. fixed environment;
5. public/OOD scope;
6. pre-registered policy;
7. successful performance promotion.

Thus a rigorous negative experiment remains valuable evidence while remaining incapable of authorizing a bad performance route.

## Result

The pre-registered M4/M5 rule is reproducibly rejected on the M24 cohort. Production routing remains conservative.
