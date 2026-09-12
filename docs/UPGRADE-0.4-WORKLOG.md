# SolverPilot 0.4 implementation

Baseline: v0.3 / 505bae7bc0d8c50a99978b6cc40fb83e37b6242d.
Research: ../research-upgrade-20260912/output/solverpilot-research-fa.html.

The implementation follows the research acceptance gates. Numerical backend
termination, independent numerical bounds and exact certificates remain separate.

## Work to complete

- [x] Reject the confirmed near-negative QP counterexamples; require actual PSD
      evidence before issuing an independent convex-QP bound.
- [x] Generalized power cone model, IR, transport, hashing and validation.
- [x] Separate matrix-first nonconvex QP/MIQP types and SCIP execution.
- [x] Explicit factorable global NLP/MINLP compilation, nonlinear equalities and
      two-sided constraints, nonsmooth supported atoms and bounded indicators.
- [x] CPU PDLP adapter for LP/diagonal convex QP, isolated native process.
- [x] Optional bounded LP witness recovery and independent certificate checks.
- [x] Conic dual reconstruction and conservative independently checked bounds;
      explicit exact PSD support limits.
- [x] Observable reuse reporting and common result evidence metadata.
- [x] Runtime capability checks for exact SCIP/GPU integrations, research-only
      learned policy with no automatic production promotion.
- [x] Documentation, examples, regression/property tests, wheel verification.
- [ ] GitHub CI validation and release preparation.

No generic global guarantee for arbitrary Python functions, no universal
infeasibility certificate, and no unmeasured GPU performance claim is introduced.

Local qualification: 1,273 collected tests, two documentation-contract mismatches repaired and rechecked; numerical suite passed; 81.11% branch-inclusive coverage. New installed-wheel run is the final local gate. Ruff correctness and typed API checks pass. Synthetic LP/OSQP timings are retained in benchmarks/results/upgrade04-local-windows.json.

Installed-wheel qualification: {'tests': '1274', 'failures': '0', 'errors': '0', 'skipped': '30'}. Current documentation/example contracts passed after updates. GitHub CI is the next gate.
