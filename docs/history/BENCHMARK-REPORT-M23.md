# Benchmark Report — M23 Production Planner

## Evidence source

M23 consumes the official-byte M22 gate:

- MIPLIB: 240 instances × 3 repetitions = 720 outcomes
- QPLIB: 19 current-IR candidates × 3 repetitions = 57 outcomes
- PACE VC public: 100 instances × 3 repetitions = 300 outcomes
- official archive/hash integrity: PASS
- fixed host / solver threads=1 / controller workers=12
- no performance-superiority claim in M22

M23 therefore classifies M22 as `corpus_validated`, not `comparative_heldout`.

## Planner campaign

900 generated canonical problems were planned: 300 LP, 300 MILP, 300 convex QP.

Results:

- completed: 900 / 900
- failures: 0
- LP selected `scipy-highs-ds`: 300 / 300
- MILP selected `scipy-highs-bridge`: 300 / 300
- QP selected `scipy-slsqp-qp-bridge`: 300 / 300
- automatic performance ranking enabled: 0 / 900
- median planning time: ~2.17 ms
- p95 planning time: ~2.74 ms
- max observed planning time: ~7.16 ms

These times are local overhead smoke measurements, not a product-wide latency guarantee.

## Solve-validation campaign

90 generated problems were actually solved through `solve_production`:

- LP: 30
- MILP: 30
- QP: 30
- independently valid solutions: 90 / 90
- MILP optimum vs exhaustive enumeration: 30 / 30 matches
- QP solution vs known analytic target: 30 / 30 matches
- failures: 0

## Performance override guard

A request to route LP to `scipy-highs-ipm` based on M22 evidence was rejected. M22 does not contain comparative held-out multi-backend evidence with feature-cost accounting. The planner retained `scipy-highs-ds`.

## Interpretation

M23 improves production safety and explainability. It intentionally does not activate learned/performance routing yet. The next evidence milestone for performance selection must benchmark multiple candidate backends on held-out groups, include planner/feature cost, and establish a nontrivial SBS→VBS opportunity before auto-routing is enabled.
