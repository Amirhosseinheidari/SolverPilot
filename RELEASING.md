# Release process

This document is the current release-process contract. Publication is manual and requires the owner-configured Trusted Publisher, protected environment approval, and a successful attested qualification of the exact main commit.

## Repository settings required before public launch

Repository owners must verify external settings that source code cannot prove:

1. the default/release branch is protected by a branch ruleset or branch protection;
2. required CI checks and pull requests are enforced; publication requires owner review through the environment;
3. GitHub **allowed-actions** policy permits only reviewed actions and current workflows use immutable full commit SHAs;
4. **GitHub Private Vulnerability Reporting** is enabled before the repository is public;
5. a protected GitHub environment named `pypi` exists with the intended reviewer protection;
6. the PyPI project has a Trusted Publisher bound to the exact repository/workflow/environment identity;
7. final project name, SPDX license expression, license file(s), maintainer/author decision, and project URLs are approved.

## Normal CI

`.github/workflows/ci.yml` runs on pull requests and pushes to `main`. It tests the source tree on Ubuntu, Windows, and macOS/Python 3.12–3.14. It is **not** release-compatibility evidence.

## Exact-artifact release qualification

`.github/workflows/release-qualification.yml` is manual-only. A release candidate is qualified as follows:

1. start from a reviewed release commit;
2. build with the pinned PEP 517 backend and pinned release tooling;
3. set `SOURCE_DATE_EPOCH` from the release commit and build twice;
4. run the reproducibility comparison;
5. run `twine check --strict`;
6. freeze SHA-256 for the single wheel/sdist pair;
7. upload the exact pair once;
8. every compatibility cell downloads and rehashes those exact bytes before installation;
9. core Linux/macOS/Windows cells run Python 3.12–3.14 in isolated environments, including `pip check`, solve smoke, and full regression;
10. a dedicated security job installs the exact wheel into an isolated target, runs pinned `pip-audit`, and emits a CycloneDX JSON SBOM;
11. Python 3.12 minimum-dependency qualification uses NumPy 2.2.0 and SciPy 1.15.0;
12. optional backend cells must make the requested backend healthy; fallback does not count; Ubuntu covers Python 3.12–3.14 and macOS/Windows cover every Python 3.12–3.14 cell;
13. when `attest_artifacts=true`, a least-privilege job creates build-provenance attestations plus an SBOM attestation and verifies artifact provenance;
14. the technical gate requires the build, security audit, core, minimum-dependency, and every native-integration matrix to pass before a run can be considered release-qualified.

A private repository/account plan that cannot use attestations may run technical qualification with `attest_artifacts=false`, but **publication requires an attested qualified run**.

## Publication

`.github/workflows/publish.yml` is manual-only and defaults to TestPyPI. The `pypi` and `testpypi` environments accept only main and require owner approval. Configure the matching pending publisher in each registry before dispatching.

Publication must:

1. receive a successful qualification `run_id` and exact 40-character `commit_sha`;
2. use the GitHub API to verify that the run succeeded for that exact commit and the release-qualification workflow;
3. download `release-dist` from that exact prior run using `github-token` + `run-id`;
4. re-verify the frozen distribution hashes and `twine check --strict`;
5. require valid GitHub attestations for the wheel and sdist;
6. publish only through PyPI Trusted Publishing/OIDC from the corresponding protected `pypi` or `testpypi` environment;
7. never use a long-lived PyPI API token;
8. record the final PyPI file hashes and provenance after publication.

## Emergency rule

Never bypass a compatibility, security, provenance, or publication gate because a release is urgent. A failed or unavailable gate delays publication; it does not become a pass.


## Protected publication environment

The `pypi` GitHub environment must have at least one **required reviewer** before any publishing workflow is enabled.
