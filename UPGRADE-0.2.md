# SolverPilot 0.2 acceptance plan

Owner scope: deliver the combined 0.1.1–0.4 roadmap as version 0.2. Preserve
the published 0.1 release. Do not claim untested capabilities or performance.

## Required work

- [x] Cache immutable package-version metadata; avoid registry creation for explicit backends.
- [x] Remove duplicate QP convexity work without weakening direct construction checks.
- [x] Expose validation tolerances through execution, automatic solving, and portfolios.
- [x] Report preparation/update/solve/validation times and original-model quality diagnostics.
- [x] Independently check LP/convex-QP optimality evidence and available termination certificates.
- [x] Reliable repeated solves, cache invalidation, isolated concurrent session ownership.
- [x] Bounded batch solving, cancellation/deadlines, cleanup, and explicit resource limits.
- [x] Accessible named/vector modeling, soft constraints, lexicographic objectives, named results.
- [x] Direct Clarabel adapter: SOC/RSOC, quadratic objectives, PSD, then verified additional cones.
- [x] Numerical, repeated-solve, memory and application benchmark campaigns with reproducible records.
- [x] Update current documentation, public API snapshots, examples and exact-artifact release gates.
- [ ] Full local regression and performance comparison; GitHub CI and release qualification.
- [ ] TestPyPI install/hash/smoke, PyPI publication and GitHub release for 0.2.

## Release boundaries

No learned selector, generic global nonconvex MINLP, or large exact TSP/VRP claims.
Original-space validation remains authoritative. Unsupported resource controls
must be rejected explicitly. Historical evidence remains historical.

## Evidence

Baseline: 0.1 has the same package files as 0.1.0rc2 except `_version.py`.
Prior audit: 1098 passed / 30 skipped; 85.0% line and 69.4% branch coverage.
Timing baseline lives outside the repository in `../performance-audit-20260911`.

Local evidence: [raw samples and report](docs/release/evidence-0.2/README.md).
GitHub qualification/publication remain pending until their external gates complete.
