# TEST REPORT M29

Version: **0.0.34**

## Full regression

- Collected: **267**
- Passed: **264**
- Skipped: **3**
- Failures: **0**
- Errors: **0**

Skipped collections are the pre-existing optional public-native integrations for `highspy`, `osqp`, and `PySCIPOpt`, which are unavailable in this runtime. They are not counted as passes.

## M29-specific verification

- Focused M29 suite: **8/8 passed**.
- Canonical positive row/objective scaling property campaign: **200/200 passed**.
- Feature extraction: **48/48 M25 + 48/48 M28** complete.
- Presolve-only probe: **48/48 + 48/48** complete.
- M25/M28 instance overlap: **0**.
- Source/protocol integrity gates: **all passed**.
- Missing/tampered opportunity context fails closed.
- M29 discovery evidence can never become production performance authority.

## Wheel

- Wheel: `optimind_core_codename-0.0.34-py3-none-any.whl`
- SHA-256: `0f2a6cc83a8d4e854f1ab171c38eeedc7e368638a01cda54b284de36cf1a0048`
- Wheel ZIP integrity: **PASS** (`testzip() = None`).
- Isolated import outside source tree: **PASS**.
- Version metadata: **0.0.34**.
- Production override with M29 evidence: **rejected**.
- Conservative LP smoke: **VALID_OPTIMAL**, independently valid.
