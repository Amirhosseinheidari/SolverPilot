# P6 STATUS — Conic Optimization Layer

**Package:** `optimind-core-codename 0.0.37`  
**Track:** P — Platform V2  
**Classification:** VERIFIED RELEASE  
**Date:** 2026-09-06

## Scope delivered

P6 adds a solver-independent conic execution layer while preserving P0–P5 trust boundaries:

- `ConicProblem` execution IR with linear rows, convex quadratic objective representation, variable bounds, and affine cone blocks
- cone kinds: second-order cone (SOC), rotated second-order cone (RSOC), positive-semidefinite cone (PSD)
- semantic `Model.soc(...)`, `Model.rotated_soc(...)`, `Model.psd(...)`
- semantic-to-ConicIR lowering with source mapping and P2 cache/re-lowering integration
- independent cone membership validation and original-space semantic validation
- P3 capability-v2 requirements for conic families
- verification-only CasADi 3.7.2 -> bundled SuperSCS bridge

## Verified solve boundary

The executable backend surface is deliberately narrower than the representational surface.

- linear-objective SOC: **VERIFIED**
- linear-objective RSOC: **VERIFIED**, via exact linear map to SOC
- PSD: IR/semantic validation **SUPPORTED**, SuperSCS solve **UNVERIFIED / FAIL-CLOSED** after negative objective-accuracy conformance
- quadratic-objective conic programs: IR representation **SUPPORTED**, SuperSCS solve **UNVERIFIED / FAIL-CLOSED** after negative conformance
- exponential/power/generalized-power cones: not implemented in P6

## Correctness evidence

- full regression: **388 collected / 385 passed / 3 skipped / 0 failed / 0 errors**
- SOC analytic campaign: **100/100**, max objective error `3.0025226616103851e-07`
- RSOC analytic campaign: **60/60**, max objective error `1.4648588742005586e-07`
- PSD IR validation: **100/100** known-feasible accepted; **100/100** known-invalid rejected
- quadratic-SOC known optimum validator: **50/50**
- quadratic-SOC SuperSCS fail-closed gate: **50/50**
- parameter/cached conic re-lowering: **100/100**, max cached/full cone offset difference `0.0`

## Performance evidence

Compiler measurements are overhead characterization only. They are not solver speed claims.

- 1 cone median compile: `0.677876 ms`
- 10 cones: `2.955872 ms`
- 50 cones: `17.053108 ms`
- 100 cones: `24.782939 ms`
- parameter full conic relower median: `0.614417 ms`
- snapshot cache hit median: `0.009830 ms`

## Deliberate boundaries

P6 does not claim Clarabel/SCS public Python integration in this runtime, SDP solving, exponential/power cones, or generic quadratic-conic solving. Representation support is not upgraded to verified backend support without adapter-specific conformance.

## Result

**P6 is VERIFIED.** Wheel isolation, negative capability boundaries and final integrity gates are recorded in `RELEASE-CHECK-P6.md` and the final re-audit.
