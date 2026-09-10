# SolverPilot public documentation — 0.1.0rc2

`0.1.0rc2` supersedes the local `0.1.0rc1` candidate after correctness/trust hardening. It packages the hardened SolverPilot Core + Track P P0–P9 platform together with the locally verified S2–S10 additive layers.

## Current public surface

- distribution: `solverpilot`
- canonical namespace: `solverpilot`
- Python: `>=3.12,<3.15`
- license: Apache-2.0
- frozen top-level API: 78 symbols (`PUBLIC-API-V0_1_0RC2.json`)
- canonical backend contract: `BACKEND-CONTRACT-V0_1_0RC2.json`

New trust/application APIs remain submodule-scoped (`solverpilot.io`, `solverpilot.evaluation`, `solverpilot.intelligence`, `solverpilot.reporting`, `solverpilot.extensions`, `solverpilot.applications`, and `solverpilot.history`).

## Publication status

This source line is a release candidate, not a publication authorization. PyPI publication remains fail-closed until the manual exact-artifact GitHub qualification matrix succeeds for the exact release commit/artifacts and the protected publishing environment/Trusted Publisher configuration is approved.

## rc2 trust hardening

The rc2 line canonicalizes accepted QP Hessians, hardens conic/non-finite validation, freezes result trust payloads, and makes reference-free benchmark exact-success proof-aware. These changes are correctness gates, not new performance claims.
