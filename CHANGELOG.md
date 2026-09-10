# Changelog

## 0.1.0rc2 — Correctness and trust hardening

- canonicalizes every accepted QP Hessian to one symmetric matrix before hashing, validation, or backend translation; pairwise symmetry checks can no longer be relaxed by an unrelated large coefficient;
- replaces raw-scale convexity tolerance with PSD-preserving local congruence scaling so a huge positive coefficient cannot hide material negative curvature;
- makes conic bound/row feasibility componentwise and hardens PSD validation against the same global-scale masking failure;
- rejects non-finite semantic/NLP candidates, objective coefficients, invalid infinity directions in bounds, and non-finite validation tolerances;
- makes runtime and CP result payloads immutable snapshots so post-solve mutation cannot manufacture stronger proof claims or desynchronize a validated candidate from its report;
- makes reference-free benchmark/oracle success proof-aware, requires protocol/environment/instance identity, and rejects non-finite timing costs;
- records solver-reported objective values for native HiGHS/OSQP paths instead of validating a value recomputed from the same candidate;
- hardens final pre-public release engineering by replacing production `assert` invariants with explicit exceptions, adding destructive CP/MPS/proof regressions, expanding optional-integration qualification across macOS/Windows Python 3.12–3.14, and gating release qualification on static correctness, dependency audit, SBOM generation, and least-privilege artifact/SBOM attestation;
- supersedes the locally audited `0.1.0rc1` source artifact; public publication remains fail-closed pending exact-artifact external qualification.

## 0.1.0rc1 — Integrated trust/application release candidate

- promotes the locally verified S2–S10 additive layers into a distinct prerelease version rather than overwriting the historical `0.0.40rc2` artifact;
- adds strict JSON/CSV ingestion and provenance, oracle/regret evaluation, versioned feature/OOD diagnostics, claim-safe reporting, explicit extension staging, TSP and CVRP/VRPTW application packs, and opt-in SQLite history;
- preserves the frozen 78-symbol top-level API by keeping new capabilities under explicit submodules;
- hardens cross-system trust boundaries around oracle comparability, report immutability, and independent optimality-proof issuance;
- retains learned LP performance routing as disabled and keeps public publication fail-closed pending exact-artifact external qualification.

## 0.0.40rc2 — Trust and numerical hardening after Track P merge

- fail closed for unverified/unknown quadratic convexity at capability, inspection, planning, runtime, and guarded backend boundaries;
- use scale-aware PSD/symmetry checks and primal feasibility validation while continuing to report raw residuals;
- separate backend-reported optimality from independent primal/dual/gap/certificate evidence through `SolveResult.optimality_evidence`;
- strengthen CP IR referential/type invariants and make parent revalidation capable of revoking an isolated worker's proof flag;
- add an independent CP controller timeout;
- record NLP `log`/`sqrt`/division domain hazards and use deterministic finite initial-point search;
- distinguish solver-certified MINLP global proof scope from `independently_verified_global`;
- reject ambiguous multiple MPS RHS/RANGES/BOUNDS vectors instead of silently selecting one;
- constrain declared Python/runtime dependency ranges to the qualification line;
- isolate OR-Tools CP-SAT native solving in a subprocess and revalidate results in the parent process to avoid a verified HiGHS shared-library ABI collision.

- forward-ports the frozen Track P P0–P9 feature line onto the hardened SolverPilot rc8 codebase instead of overwriting M27–M33/release-hardening changes;
- adds the semantic modeling kernel, parameter compiler/cache, Capability Protocol 2.0, bridge/transformation-tape engine, persistent sessions, conic layer, smooth NLP/AD, certified convex binary MINLP orchestration, and CP/CP-SAT core;
- renames all live Track P Python modules/schema identifiers/metadata to the `solverpilot` namespace while preserving original OptiMind naming only in frozen history;
- keeps the M30-frozen 78-symbol top-level API unchanged; Track P APIs are canonical under explicit submodules;
- adds `conic`, `nlp`, `minlp`, and `cp` optional extras and the `solverpilot-capabilities` CLI;
- expands README, examples, API docs, and limitations around the exact guarantee boundaries of the merged functionality;
- retains the rejected learned LP performance selector as disabled; no new performance-ranking claim is introduced by this merge.

## 0.0.36rc7 — Final pre-GitHub release cleanup

- removes the pre-public `optimind` compatibility package to eliminate nested-import class identity duplication;
- removes the internal `OptiMindError` compatibility alias from live source;
- makes release manifest/smoke validation derive current name/version/Python floor from `pyproject.toml`;
- derives RC API/backend snapshot filenames from the current prerelease version;
- makes release smoke fail if a legacy `optimind` namespace is shipped;
- moves historical M31/M32/M33 release helper scripts out of live `tools/`;
- refreshes pre-GitHub hardening/verification documentation.

## 0.0.36rc6 — SolverPilot brand and package migration

- renames the public brand, distribution, CLI entry points, and canonical Python namespace to SolverPilot / `solverpilot`;
- records author/maintainer metadata;
- creates SolverPilot API/backend contract snapshots while preserving historical backend IDs;
- retains the old OptiMind naming only as historical research provenance.

## 0.0.36rc5 — Apache-2.0 license decision

- adopts Apache License 2.0 (`Apache-2.0`);
- adds the root `LICENSE` and PEP 639 license metadata;
- keeps publication fail-closed pending final repository/service configuration.

## 0.0.36rc4 — Pre-GitHub hardening

- fixes atomic benchmark reacquisition, download verification semantics, and relative verification paths;
- separates lightweight PR CI from manual exact-artifact release qualification;
- binds disabled publishing to an exact successful qualification run/commit;
- removes release binaries from the source tree and adds repository security/contribution policy files.

## 0.0.36rc3 — M33 External CI & Publication Decision

- narrows intended 1.0 support to CPython 3.12–3.14 and adds Python 3.14 to the exact-artifact matrix;
- raises runtime floors to NumPy 2.2 and SciPy 1.15, with a dedicated minimum-dependency regression;
- runs compatibility cells inside isolated virtual environments;
- adds PyPA pip-audit runtime dependency checks and CycloneDX evidence;
- documents the release process per current Scientific Python/GitHub/PyPI security guidance;
- adds Dependabot for reviewed GitHub Action SHA updates and a disabled OIDC-only PyPI publication template;
- adds same-source reproducibility diagnostics;
- keeps public RC/1.0 authorization fail-closed until external execution and owner metadata complete.


## 0.0.36rc2 — M32 External Compatibility & Supply-Chain Hardening

- preserves the M30/M31 frozen 78-symbol public API and backend IDs;
- changes CI to build-once/test-many using the exact release wheel;
- records resolver NumPy/SciPy versions and requires `pip check` in every compatibility cell;
- pins all external GitHub Actions to immutable full commit SHAs;
- adds build-provenance attestation support while leaving publication disabled;
- keeps public RC/1.0 fail-closed until the complete external matrix and owner publication metadata are available.


## 0.0.36rc1 — M31 Technical Release Candidate Compatibility Checkpoint

- froze M30 public API/backend contract into M31 snapshots;
- added a fail-closed 3-OS x 4-Python compatibility workflow;
- added Python-3.10 grammar audit and current Linux/Python-3.13 execution evidence;
- separated upstream dependency-wheel availability from SolverPilot compatibility claims;
- recorded unresolved license/name/maintainer/URL owner decisions and public-brand collision risk;
- public 1.0 compatibility remains unauthorized until CI matrix execution.


## 0.0.35 — M30 Release Consolidation

- Freeze a stable top-level public API for the 1.0 release line.
- Move milestone-specific evidence translators and the rejected selective-LP evaluator to `solverpilot.experimental`.
- Add a stable public exception hierarchy for backend availability, capability mismatch, and planner routing failures.
- Replace stale research-oriented README content with a product-facing scope/claim boundary.
- Add explicit release/provenance/API documentation and public-API manifest tests.
- Remove generated `egg-info`, bytecode caches, and stale root-level milestone clutter from the release source tree.
- Keep learned LP performance routing disabled after M24–M29 negative generalization evidence.

## 0.0.34 — M29

- Value-of-information/feature-representation audit for LP DS/IPM selection.
- No feature family passed cross-cohort authorization; learned LP routing closed for the 1.0 line.

Earlier milestone detail is preserved under `docs/history/` and in the complete benchmark archive.
