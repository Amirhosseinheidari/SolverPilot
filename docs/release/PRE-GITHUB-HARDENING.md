# Pre-GitHub Hardening — 0.1.0rc2

This document is the **current local hardening state** for SolverPilot. Historical rc4–rc8, `0.0.40rc2`, and `0.1.0rc1` material is preserved under `docs/history/` and is provenance only.

## Current identity

- brand: **SolverPilot**
- distribution: `solverpilot`
- canonical Python namespace: `solverpilot`
- version: `0.1.0rc2`
- license: Apache-2.0
- author/maintainer: Amirhossein Heidari Rashtabad
- Python target: CPython 3.12–3.14
- intended qualification OS matrix: Linux / macOS / Windows

## Correctness/trust hardening closed in rc2

`0.1.0rc2` supersedes the local `0.1.0rc1` candidate after adversarial pre-public review found trust-boundary defects that ordinary regression tests had not exposed.

- QP Hessians are pairwise symmetry-checked, canonicalized to one symmetric matrix, and convexity-tested after PSD-preserving local scaling. A large unrelated coefficient can no longer hide asymmetry or material negative curvature.
- Native HiGHS/OSQP adapters preserve a solver-reported objective for independent objective-consistency checking where the solver exposes it.
- Conic variable/linear validation is componentwise; PSD validation uses local symmetry checks and congruence scaling rather than one global matrix norm.
- Semantic and NLP validators reject non-finite candidates/objectives before arithmetic can turn NaN/Inf into a false pass.
- Linear/QP objective vectors must be finite; lower bounds may not be `+inf` and upper bounds may not be `-inf`; validation tolerances must be finite and non-negative.
- `SolveResult` and CP result payloads are immutable snapshots. Candidate arrays, assignments, nested trust statistics, and derived optimality evidence cannot be mutated after validation.
- Reference-free exact benchmark/oracle success requires explicit independent optimality evidence; benchmark timing aggregation requires protocol/environment/instance identity and finite non-negative cost data.

The counterexamples for these defects are retained as permanent adversarial regression tests.

## Release-tool contract

Current release validation derives the distribution name, version, Python floor, API snapshot filename, backend-contract filename, audit filename, and public-doc filename from `pyproject.toml`. It must fail if wheel/sdist identity differs from the current source, if a legacy `optimind` package is shipped, or if required release documents are absent.

The manual release-qualification workflow remains build-once/test-many. Every compatibility cell must download and verify the exact artifacts before installation.

## Remaining external blockers

The source tree cannot prove or configure these repository/service states by itself:

- final GitHub repository/docs/issues URLs;
- branch ruleset/protection and required checks;
- Private Vulnerability Reporting and allowed-actions policy;
- protected `pypi` environment and required reviewer;
- PyPI Trusted Publisher identity;
- exact-artifact qualification on Linux/macOS/Windows with Python 3.12–3.14 and required optional backends;
- artifact attestation verification for the exact release commit.

These remain fail-closed publication gates.

## Known design debt

High-level solve functions still do not expose per-call `ValidationTolerances`. Infeasible/unbounded public statuses normally represent backend conclusions unless independent certificate/diagnostic evidence is explicitly recorded. Learned LP performance routing remains disabled.
