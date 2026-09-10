# M32 External Compatibility & Publication Closure Protocol

**Frozen before M32 implementation changes.**

Baseline: M31 Final Verified archive SHA-256 `55928bd7084ae69862b1385b5e06998bb41f19df181ad7bf5fd1aa4629f79818`.

Intended package prerelease for M32 hardening: `0.0.36rc2`.

## 1. Purpose

M32 determines whether the M31 technical release candidate can be promoted toward a **public release candidate**. It is a release-validation milestone, not a solver-selection or performance-research milestone.

M32 must distinguish four different forms of evidence:

1. **executed compatibility evidence** — the exact release artifact was installed and exercised on the named OS/Python environment;
2. **static compatibility evidence** — syntax/metadata analysis only;
3. **upstream artifact availability** — dependency wheels exist for a platform/interpreter;
4. **planned CI coverage** — workflow definition exists but has not executed.

Only (1) can turn a compatibility matrix cell into `executed_pass`.

## 2. Claim boundary

M32 must never:

- mark a pending OS/Python cell as pass because a dependency wheel exists;
- infer Python 3.10/3.11 compatibility from `requires-python >=3.10` alone;
- test source checkouts in matrix cells and call that exact-release-artifact validation;
- replace a failed resolver/install with a different undeclared dependency set and call it equivalent;
- publish to TestPyPI/PyPI while license/name/owner metadata are unresolved;
- enable learned LP performance routing;
- convert syntax-only Python 3.10 parsing into runtime support evidence.

## 3. Core compatibility matrix

Target claim set remains:

- OS: Linux, macOS, Windows
- CPython: 3.10, 3.11, 3.12, 3.13

Total core cells: 12.

A core cell passes only if it:

1. receives the **same wheel artifact built once** by the release build job;
2. verifies the wheel SHA-256 against the build manifest;
3. installs that wheel with dependencies resolved by pip for that Python;
4. records exact Python, pip, NumPy and SciPy versions;
5. runs `pip check` successfully;
6. executes LP, MILP and convex-QP smoke tests with independent validation;
7. verifies conservative production routing (`scipy-highs-ds`, learned performance ranking off);
8. runs the full regression suite with zero failures/errors;
9. uploads JUnit and resolver/environment evidence.

## 4. Dependency-resolution gate

The project currently declares:

- `numpy>=1.24`
- `scipy>=1.10`

This intentionally leaves pip free to choose an interpreter-compatible release. M32 records the actual resolver result per matrix cell.

Important current ecosystem fact: the newest NumPy/SciPy releases no longer support every Python covered by the project floor. Therefore M32 tests the resolver output rather than assuming that “latest dependencies” are compatible with Python 3.10/3.11.

Public Python support can only be claimed for cells whose resolved dependency set installs and passes the package tests.

## 5. Optional backend matrix

Target optional integrations:

- `highspy>=1.15.1,<2`
- `osqp>=1.1.3,<2`
- `pyscipopt>=6.2.1,<7`
- `nlopt>=2.11,<3`
- `casadi==3.7.2` (verification bridge)

For an optional backend to pass on a cell:

- the exact OptiMind release wheel must already be installed;
- the declared optional dependency version must be installed;
- backend-health must report the intended backend as healthy;
- a backend-specific solve must execute; silent fallback to a different backend is forbidden.

Upstream PyPI wheel availability is recorded separately from execution status.

## 6. Build-once / test-many artifact gate

M32 changes the CI topology from “install source in every matrix cell” to:

1. build one wheel + one sdist;
2. validate package metadata/content;
3. compute and publish artifact SHA-256;
4. upload the exact artifacts;
5. downstream matrix jobs download those artifacts;
6. test the exact wheel bytes;
7. independently rebuild the wheel from the sdist and compare functional smoke behavior.

The build job is the sole source of release artifacts.

## 7. Supply-chain gate

All third-party GitHub Actions used by M32 must be pinned to a full 40-character commit SHA. Mutable tags such as `@v7` are forbidden in the M32 release workflow.

The release design should support:

- build provenance attestations for distributions;
- artifact digests;
- Trusted Publishing / OIDC rather than long-lived PyPI API tokens when publication is eventually authorized;
- PyPI digital attestations where supported.

M32 may prepare these mechanisms but must not publish before governance metadata is complete.

## 8. Package metadata/publication gate

Public RC authorization requires owner-provided values for:

1. SPDX license expression and matching license file(s);
2. final public distribution/project name;
3. author/maintainer metadata if desired;
4. source/documentation/issue/homepage URLs.

M32 must not invent these fields.

The public-facing name `OptiMind` also requires owner review because the name is already used publicly by a Microsoft/OptiGuide optimization-reasoning research project. This is a branding-review flag, not a legal conclusion.

## 9. Packaging validation gate

Before an M32 artifact can be called technically publishable:

- wheel ZIP integrity must pass;
- sdist archive integrity must pass;
- package version metadata must equal source version;
- wheel must contain no tests/build cache/source egg-info pollution;
- sdist must contain the frozen public API/backend contracts and release documentation;
- sdist -> wheel -> isolated install -> smoke must pass;
- `twine check` should pass when `twine` is available; if unavailable due environment/network, this remains explicit evidence gap rather than an implicit pass.

## 10. Public RC promotion gate

`public_rc_authorized = true` requires **all** of:

- 12/12 core compatibility cells `executed_pass`;
- zero core test failures/errors;
- exact-release-artifact build-once/test-many contract satisfied;
- optional integrations required by the advertised extras pass their intended matrix;
- package metadata/governance fields complete;
- final wheel/sdist integrity and isolated installation pass;
- supply-chain workflow uses immutable action SHAs;
- learned LP performance routing remains off unless independently authorized by its evidence contract.

Anything less yields `public_rc_authorized = false`.

## 11. Literature / standards basis

M32 protocol is based on current primary documentation from:

- Python Packaging User Guide / Core Metadata / PEP 639;
- GitHub Actions Python testing and secure-use guidance;
- PyPI Trusted Publishing and digital attestation documentation;
- PyPA official PyPI publish action;
- current PyPI metadata/wheels for NumPy, SciPy and optional solver bindings.

Full dated URLs and findings are recorded in `INTERNET-AUDIT-M32.md`.
