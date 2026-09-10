# Test Report — M26

## New M26 checks

- exact 48-instance split coverage
- group leakage rejection / group-safe frozen split
- 20/12/16 train/validation/test counts
- training artifact contains zero test rows/names
- model reports no test outcomes seen during fit
- all 9 model candidates worse than SBS is recorded, not hidden
- validation gate failure prevents production performance authority
- failed-selector performance override is rejected
- sealed-test result artifacts are absent

## Release verification

A final full regression and isolated wheel smoke are required before 0.0.31 is classified as verified. See `RELEASE-CHECK-M26.md` and verification artifacts.
