# BENCHMARK REPORT M29

## Purpose

M29 asks whether richer LP representations contain enough **stable, cost-effective, cross-cohort information** about `delta_s = IPM cost - simplex cost` to justify one final fresh-corpus selector experiment.

M29 is discovery-only because the M25 and M28 outcomes were known before the feature audit.

## Protocol

Exact pre-registered protocol: `docs/research/M29-VALUE-OF-INFORMATION-PROTOCOL.md`.

Cohorts:

- M25: 48 MIPLIB-derived public/OOD LP relaxations, robust targets from two repeated campaigns.
- M28: 48 fresh MIPLIB-derived public/OOD LP relaxations, robust targets from three native-HiGHS rounds.
- overlap: 0.

Feature families:

- A: base size/density/aspect features.
- B: canonical normalized matrix/objective/RHS/bound/sparsity statistics.
- C: topology/decomposition proxies.
- D: bounded HiGHS presolve-only probe; no simplex/IPM trajectory features.

## Representation correction audit

The final extractor follows MIPLIB 2017's representation principle: each constraint row and its finite sides are normalized by the row max-absolute coefficient; the objective is normalized by its max-absolute coefficient. A 200-case randomized property campaign verifies invariance to positive independent row scaling and positive objective scaling:

- passed: **200/200**.
- max relative difference: **5.871e-16**.

## Feature extraction accounting

- M25: 48/48 static features, 48/48 presolve probes.
- M28: 48/48 static features, 48/48 presolve probes.
- hard feature-worker errors: 0 in final evidence.

## Stable univariate result

Only `C:row_degree_max_over_mean` passes the pre-registered stability definition. It is substantially attenuated after controlling for base size/density, especially on M25 (residual rho ≈ -0.011).

## Transfer result

| m25→m28 | A | -0.066 | 0.772 | 1.216 | 1.217 |
| m28→m25 | A | -0.053 | 0.371 | 1.330 | 1.330 |
| m25→m28 | AB | 0.251 | 0.772 | 1.054 | 1.126 |
| m28→m25 | AB | 0.085 | 0.553 | 1.255 | 1.332 |
| m25→m28 | ABC | 0.275 | 0.772 | 1.054 | 1.161 |
| m28→m25 | ABC | 0.074 | 0.553 | 1.255 | 1.361 |
| m25→m28 | ABCD | 0.135 | 0.741 | 1.115 | 1.500 |
| m28→m25 | ABCD | 0.057 | 0.537 | 1.255 | 1.506 |

No family passes the two-direction transfer gate. Even the zero-feature-cost counterfactual is worse than SBS in every direction/family.

## Feature cost finding

A is negligible (~0.06 ms median). Richer Python-side features are not negligible:

- A+B: median ≈ 7.22 ms M25 / 8.51 ms M28.
- A+B+C: ≈ 9.26 / 15.12 ms.
- A+B+C+D: ≈ 23.18 / 57.22 ms.

However, because transfer still loses to SBS when feature cost is set to zero, optimizing the extractor alone cannot rescue the current representation.

## Target stability / concentration

M25 repeated delta ranking is highly stable (rho 0.936); M28's three round-delta rankings are also stable (median rho 0.922). Opportunity is concentrated but not single-instance-only.

## Decision

No family passes all authorization gates. `m30_authorized=false`; M29 closes learned LP routing for the 1.0 release line.

## Final software verification

The final `0.0.34` source tree passed **267** tests (**264 passed, 3 skipped, 0 failures, 0 errors**). The three skips are the pre-existing optional public-native integration collections for `highspy`, `osqp`, and `PySCIPOpt` unavailable in this runtime.

The final wheel was installed into an isolated `/tmp` target and imported outside the source tree. M29 evidence remained fail-closed, an attempted LP performance override was rejected, and the conservative LP route produced an independently valid optimum.

Wheel SHA-256: `0f2a6cc83a8d4e854f1ab171c38eeedc7e368638a01cda54b284de36cf1a0048`.

