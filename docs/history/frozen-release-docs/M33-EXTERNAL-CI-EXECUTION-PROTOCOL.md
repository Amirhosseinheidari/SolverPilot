# M33 External CI Execution & Publication Decision Protocol

**Frozen before M33 implementation changes.**

Baseline: M32 Final Verified archive SHA-256 `c39c19bb123c220f43063287758b3ab979b364fd26a27458e117d2d513065d07`.

Intended M33 prerelease: `0.0.36rc3`.

## 1. Purpose

M33 is the final technical release-policy convergence milestone before a public release candidate can be authorized. It does not add solver algorithms or reopen learned LP routing. It hardens the exact release artifact, compatibility scope, minimum dependency policy, vulnerability/SBOM checks, release governance, and external execution gate.

M33 may close as a verified engineering milestone even when `public_rc_authorized=false`; pending external CI or owner/governance inputs must never be converted to pass.

## 2. Evidence classes

M33 distinguishes:

1. **executed exact-artifact compatibility** — the exact release wheel was installed and exercised on the named OS/Python cell;
2. **minimum-dependency execution** — exact declared minimum runtime dependencies were installed and the full regression passed;
3. **static compatibility** — syntax/metadata analysis only;
4. **upstream wheel availability** — a dependency publishes a candidate wheel;
5. **workflow readiness** — CI code exists but did not execute.

Only (1) can mark an OS/Python core cell `executed_pass`. Only (2) can validate the declared minimum runtime dependency floor.

## 3. Public Python support policy

M33 changes the intended public 1.0 interpreter support set to:

- CPython 3.12
- CPython 3.13
- CPython 3.14

and the core OS set remains:

- Linux
- macOS
- Windows

Total core compatibility cells: **9**.

This is a forward-looking support-policy change, not post-hoc gate relaxation: M32 had zero failed core cells and zero executed-pass core cells; all twelve M32 cells were pending external execution.

Rationale:

- Python 3.10 is near the end of its upstream security-support lifespan in October 2026.
- Scientific Python SPEC 0 recommends dropping Python versions three years after release and already places Python 3.11 outside its recommended support window.
- Current NumPy and SciPy releases require Python >=3.12.
- the principal optional solver bindings used by OptiMind publish CPython 3.14 wheels on major desktop OS families.

M33 does not claim formal adoption of every Scientific Python SPEC recommendation; it uses these ecosystem facts to define a maintainable 1.0 support surface.

## 4. Minimum runtime dependency policy

M33 raises the intended public 1.0 runtime floors to:

- `numpy>=2.2`
- `scipy>=1.15`

These versions remain inside the current SPEC 0 two-year support window and materially reduce untested/stale dependency surface relative to the prior `numpy>=1.24` / `scipy>=1.10` floors.

A dedicated Linux / Python 3.12 minimum-dependency job must install exactly the declared floors and run:

- `pip check`;
- release smoke;
- full regression with zero failures/errors.

Latest-resolver core cells remain separate and record their actual resolved versions.

## 5. Exact-artifact core matrix

Each of the 9 core cells passes only if it:

1. receives the same build-once wheel artifact;
2. verifies SHA-256 against the build manifest;
3. creates an isolated virtual environment;
4. installs the exact wheel plus test dependencies inside that environment;
5. records Python, pip, NumPy and SciPy versions;
6. passes `pip check`;
7. runs LP, MILP and convex-QP smoke with independent validation;
8. verifies conservative production routing and learned performance routing off;
9. runs the full regression suite with zero failures/errors;
10. uploads JUnit, resolver, smoke, and security evidence.

Source checkout installation is not exact-release-artifact validation.

## 6. Optional backend matrix

Advertised optional integrations remain:

- Highs / `highspy>=1.15.1,<2`
- OSQP / `osqp>=1.1.3,<2`
- SCIP / `pyscipopt>=6.2.1,<7`
- NLopt / `nlopt>=2.11,<3`
- CasADi verification bridge / `casadi==3.7.2`

M33 targets:

- Linux on Python 3.12 / 3.13 / 3.14 for all five extras;
- Linux/macOS/Windows on Python 3.14 for all five extras.

A backend cell passes only if backend health reports the requested backend healthy and a backend-specific solve executes without fallback.

Upstream wheel availability is never execution evidence.

## 7. Dependency security and SBOM gate

M33 adds a runtime dependency audit using PyPA `pip-audit`.

For each executed core cell:

1. exact installed runtime dependency versions are written to a requirements-style evidence file;
2. `pip-audit` audits those runtime requirements against a vulnerability service;
3. CycloneDX JSON output is retained as an SBOM/security artifact;
4. any known vulnerability causes the cell to fail unless an explicit, versioned, time-bounded exception with vulnerability identifier and rationale is committed before the release decision.

No vulnerability ignore list is pre-authorized by this protocol.

Audit-tool dependencies themselves are not treated as OptiMind runtime dependencies.

## 8. Supply-chain / release-process gate

M33 follows the current Scientific Python SPEC 8 / GitHub/PyPI release-security direction:

- document the complete release process in `RELEASING.md`;
- global workflow permissions default to `contents: read`;
- privileged release/publish permissions are job-local;
- all third-party Actions are pinned to full 40-character commit SHAs;
- add Dependabot configuration for reviewed GitHub Action updates;
- release publication must be intentional (`workflow_dispatch`) and use a protected GitHub environment when publication is enabled;
- use OIDC / PyPI Trusted Publishing rather than long-lived API tokens;
- generate and verify artifact attestations for released wheel/sdist;
- retain artifact SHA-256 and provenance evidence.

Repository settings such as branch protection, environment required reviewers, and allowed-actions policy are external state and must be verified from the actual repository before public RC authorization.

## 9. Build reproducibility diagnostic

M33 performs a controlled same-source double-build with a fixed `SOURCE_DATE_EPOCH`.

- byte-identical wheel/sdist hashes are recorded if achieved;
- if archives are not byte-identical, member-level differences are recorded;
- lack of full reproducibility is not by itself a public-RC failure, but it may not be falsely reported as reproducible.

## 10. Packaging/publication gate

Before public RC authorization:

- wheel and sdist integrity pass;
- `twine check --strict` passes in external build CI;
- sdist -> wheel -> isolated install -> smoke passes;
- public API/backend contracts remain frozen;
- package version metadata matches source;
- license/SPDX expression and matching license file(s) are owner-provided;
- final public distribution/project name is owner-approved;
- author/maintainer metadata decision is explicit;
- source/docs/issues/homepage URLs and Trusted Publisher identity are configured;
- branding review for the name `OptiMind` is resolved by the owner.

M33 must not fabricate owner/legal metadata.

## 11. Publish workflow gate

M33 may prepare a disabled publication template, but must not enable actual PyPI/TestPyPI upload while owner metadata or external compatibility gates are incomplete.

When enabled later, publication must use:

- GitHub Actions environment with release approval policy;
- `id-token: write` only on the publish job;
- PyPI Trusted Publishing / OIDC;
- official PyPA publishing action pinned to a full commit SHA;
- digital attestations / publish provenance supported by PyPI.

Long-lived `PYPI_API_TOKEN` secrets are forbidden by the intended release path.

## 12. Public RC promotion gate

`public_rc_authorized=true` requires all of:

- 9/9 core compatibility cells `executed_pass`;
- minimum-dependency job pass;
- zero core regression failures/errors;
- required optional backend matrix pass for every advertised extra;
- runtime vulnerability audit pass with no undocumented ignores;
- exact-artifact build-once/test-many contract satisfied;
- `twine check --strict` pass;
- wheel/sdist integrity and round-trip pass;
- action SHA pinning and attestation generation/verification pass;
- branch/environment/action repository security settings verified;
- owner publication metadata and naming decisions complete;
- learned LP performance routing remains off unless independently authorized by its own evidence contract.

Anything less yields `public_rc_authorized=false`.

## 13. Primary standards/literature basis

M33 protocol was informed before implementation by current primary material from:

- Scientific Python SPEC 0 — minimum supported dependencies;
- Scientific Python SPEC 8 — securing the release process;
- CPython 3.10 lifecycle / PEP 619;
- current NumPy/SciPy PyPI metadata;
- current optional solver wheel metadata on PyPI;
- GitHub secure-use guidance and artifact attestation documentation;
- PyPI Trusted Publishing and PEP 740 digital attestation documentation;
- PyPA `pip-audit` documentation.

Dated URLs and observations are preserved in `INTERNET-AUDIT-M33.md`.
