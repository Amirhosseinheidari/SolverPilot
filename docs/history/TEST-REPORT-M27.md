# Test Report — M27 0.0.32

## Full regression

The final suite was executed in three deterministic non-overlapping file shards.

- collected: **252**
- passed: **249**
- skipped: **3**
- failures: **0**
- errors: **0**

The three skips are the existing optional public integrations:

- `highspy` native integration
- `osqp` native integration
- `PySCIPOpt` native integration

Evidence:
- `verification/2026-09-06/m27-pytest-shard1.xml`
- `verification/2026-09-06/m27-pytest-shard2.xml`
- `verification/2026-09-06/m27-pytest-shard3.xml`
- `verification/2026-09-06/m27-regression-summary.json`

## New M27 regression coverage

M27 tests verify:

- the selector was frozen before the held-out test,
- exact 16-instance test membership is retained,
- no test costs were seen during model fitting,
- the serialized low-capacity selector can run without a scikit-learn runtime dependency,
- non-LP inputs are rejected by the research selector,
- the real held-out test is recorded as consumed and negative,
- real M27 evidence does not authorize performance ranking,
- manual performance overrides remain rejected,
- model/protocol hashes in the held-out result match the frozen artifacts,
- tampering with the frozen model invalidates held-out integrity,
- absent test-time environment provenance cannot be silently promoted to `fixed_environment=true`.

## Installed-wheel smoke

The 0.0.32 wheel was installed into an isolated target outside the source tree. From that installation:

- version/import location: PASS
- M27 evidence parses as `comparative_heldout`: PASS
- `heldout = true`: PASS
- selection opportunity remains true: PASS
- performance ranking remains false: PASS
- `supports_performance_ranking = false`: PASS
- manual IPM speed override is rejected: PASS
- conservative production LP selection remains enabled: PASS
- an LP solve returns `VALID_OPTIMAL`, objective 1.0, independently valid: PASS

Wheel smoke: `verification/2026-09-06/m27-wheel-smoke.json`.

## Final audit

A separate post-hoc integrity audit verifies frozen hashes, split leakage, M22/M25 lineage, solver-status accounting, and the conservative environment-provenance downgrade. See `benchmarks/results/m27/m27-posthoc-integrity-audit.json`.
