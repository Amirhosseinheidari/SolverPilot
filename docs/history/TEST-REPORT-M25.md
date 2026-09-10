# Test Report — M25 / 0.0.30

## Full regression

The full test inventory was deterministically distributed across three non-overlapping file shards.

- tests: **234**
- passed: **234**
- skipped: **0**
- failures: **0**
- errors: **0**

Evidence:
- `verification/2026-09-06/m25-shard1.xml`
- `verification/2026-09-06/m25-shard2.xml`
- `verification/2026-09-06/m25-shard3.xml`
- `verification/2026-09-06/m25-regression-summary.json`

## New M25 tests

M25 adds regression coverage proving:

- positive opportunity evidence is represented separately from selector authority,
- positive M25 opportunity evidence cannot enable performance ranking by itself,
- an unstable/negative opportunity record is not validated,
- the real two-run M25 stability artifact parses as positive opportunity evidence only,
- valid synthetic comparative selector evidence must now also declare `selection_opportunity_validated=True` before it can authorize ranking.

## Wheel smoke

From an isolated wheel install outside the source tree:

- version 0.0.30: PASS
- M25 real evidence -> `selection_opportunity_validated=True`: PASS
- M25 real evidence -> `performance_ranking_validated=False`: PASS
- `supports_performance_ranking=False`: PASS
- attempted IPM override rejected: PASS
- selected production LP backend remains `scipy-highs-ds`: PASS
- production LP solve returns independently valid optimum: PASS
