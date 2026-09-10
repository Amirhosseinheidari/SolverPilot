# Test Report — M23

## Full regression

- collected: 229
- passed: 226
- skipped: 3
- failures: 0
- errors: 0

The three skips are the existing optional public native integrations unavailable in this runtime.

JUnit: `verification/2026-09-06/m23-pytest-junit.xml`

## M23-specific tests

Eight production-planner tests verify:

1. M22 evidence is classified as corpus-validation, not comparative performance evidence.
2. LP and MILP use conservative HiGHS baselines.
3. M22 evidence rejects speed overrides.
4. A synthetic valid comparative-heldout contract can override only an eligible backend.
5. `PROVE_OPTIMAL` rejects non-certifying QP routes.
6. balanced QP can use a non-certifying feasible route without mislabeling performance evidence.
7. `solve_production` returns an independently validated solution and auditable plan.
8. the real M22 gate JSON parses to corpus-validated evidence while keeping performance ranking disabled.

## Installed-wheel smoke

From an isolated wheel installation outside the source tree:

- version: 0.0.28
- real M22 gate parsed: `corpus_validated`
- performance ranking from M22: false
- LP route: `scipy-highs-ds`, `valid_optimal`
- MILP route: `scipy-highs-bridge`, `valid_optimal`, objective 1.0
- QP route: `scipy-slsqp-qp-bridge`, independently valid feasible solution near [1,2]
- QP proof-intent route: correctly rejected because available QP routes explicitly lack optimality certificates
