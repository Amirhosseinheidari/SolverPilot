# P4 Status — Bridge / Reformulation Engine + Transformation Tape

**Version:** 0.0.35  
**Classification:** VERIFIED P4 RELEASE  

## Completed

- Semantic `IndicatorConstraint` preserved before lowering.
- Deterministic bridge registry and safe bridge policy.
- P3 capability-aware native-path reasoning.
- Fixed-active and fixed-inactive exact indicator simplifications.
- Finite-bound-certified affine Big-M bridge for <=, >= and == bodies, active value 0 or 1.
- Fail-closed rejection when finite Big-M cannot be certified.
- Versioned precondition certificates and SHA-256 certificate identity.
- Transformation Tape with exactness, mappings, size delta and numerical-risk metadata.
- Source-map rows and generated target IDs.
- Plan-sensitive parameter dependencies and bridge certificate revalidation.
- Original-space semantic primal validation.
- Transformation Tape JSON Schema packaged in the wheel.
- Legacy P2 compile/cache path preserved for models without semantic bridges.

## Verification summary

- Full regression: 348 collected / 345 passed / 3 skipped / 0 failed / 0 errors.
- P4 finite-domain semantic equivalence/property suite: PASS.
- 100 random indicator MILP brute-force optimum cross-checks: 100/100 PASS.
- Maximum objective error in brute-force campaign: 3.552713678800501e-15.
- Plan-sensitive parameter revalidation campaign: 200/200 cached-vs-full equivalent.

## Deliberate boundaries

P4 v1 is the bridge engine plus the first exact production bridge family. PWL/SOS/general logical/conic bridge packs and native backend indicator transport are not claimed in this release.
