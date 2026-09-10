# TEST REPORT M28

Version: **0.0.33**

## Full regression

- Tests: **259**
- Passed: **256**
- Skipped: **3**
- Failures: **0**
- Errors: **0**

The three skips are the pre-existing optional public native integration tests unavailable in this runtime.

## M28-specific checks

Focused M28 suite validates:
- SciPy ambiguous `method="highs"` is not exposed by `ScipyHighsLPBackend`;
- local SciPy 1.17 semantics probe is fail-closed;
- fresh cohort has zero overlap with M25 selected rows;
- protocol/runner/worker SHA-256 provenance matches raw campaign evidence;
- 432 native outcomes are fully accounted;
- native `choose` is not mislabeled as a DS/IPM selector;
- invalid candidates remain rejected;
- M28 production evidence cannot authorize performance ranking.

## Wheel

- SHA-256: `36cc819e7c97fc5c6541bd5bb942903c8ff3f9c004f35b8c02b57fbc484a7392`
- wheel ZIP integrity: `None`
- metadata version: `0.0.33`
- isolated source-tree import: `False`
- isolated LP status: `valid_optimal` / valid `True`
- performance ranking support: `False`
