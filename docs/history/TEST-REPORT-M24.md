# Test Report — M24

**Version:** 0.0.29

## Full regression

- collected: **233**
- passed: **230**
- skipped: **3**
- failed: **0**
- errors: **0**

JUnit: `verification/2026-09-06/m24-pytest-junit.xml`.

## M24-specific contract tests

M24 adds tests proving that:

- negative public/OOD comparative evidence remains `COMPARATIVE_HELDOUT` but cannot authorize performance ranking;
- a failed M24 selector rejects an explicit `scipy-highs-ipm` speed override;
- production performance ranking requires public/OOD + preregistration + successful validation/promotion in addition to the M23 held-out/fixed-environment/feature-cost conditions;
- the retained real M24 result has 24 instances, zero objective mismatches, DS-dominant outcome, and `selector_promoted=false`.

Existing M23 positive synthetic unit evidence was updated to set all strengthened authorization flags explicitly, so the success path is still tested without weakening the new gate.
