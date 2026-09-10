# M32 Publishing Plan — Disabled Until Owner Authorization

M32 prepares a secure publication path but does **not** activate PyPI/TestPyPI publishing while project-owner metadata is unresolved.

## Intended secure flow after authorization

1. build wheel + sdist exactly once;
2. run package metadata/integrity checks;
3. test the exact wheel across the declared compatibility matrix;
4. generate artifact SHA-256 and GitHub build provenance attestation;
5. optionally validate on TestPyPI using the final public project name;
6. publish to PyPI using **Trusted Publishing / OIDC**, not a stored long-lived API token;
7. allow the official PyPA publish action to generate/upload PyPI digital attestations;
8. archive the final distributions, attestations and manifest.

## Supply-chain pins selected for M32

The active M32 compatibility workflow pins third-party GitHub Actions to full commit SHAs:

- `actions/checkout` v7.0.1: `3d3c42e5aac5ba805825da76410c181273ba90b1`
- `actions/setup-python` v7.0.0: `5fda3b95a4ea91299a34e894583c3862153e4b97`
- `actions/upload-artifact` v7.0.1: `043fb46d1a93c77aae656e7c1c64a875d1fc6a0a`
- `actions/download-artifact` v8.0.1: `3e5f45b2cfb9172054b4087a40e8e0b5a5461e7c`
- `actions/attest` v4.2.1: `508db95dd578ae2727ebd6217d5ba78e4fbda05d`

GitHub's secure-use documentation states that a full-length commit SHA is the only immutable way to reference an action.

## Future PyPI action

When owner metadata is complete, the recommended publisher is the PyPA-maintained `pypa/gh-action-pypi-publish` using Trusted Publishing. The current action documentation explicitly recommends OIDC trusted publishing and warns against mutable branch pointers. M32 does not add an active publisher job because doing so before the final project name/license/environment are known would create a misleading release path.

## Required owner decisions before activation

- final public distribution/project name;
- SPDX license expression + license file(s);
- author/maintainer identity if desired;
- repository/docs/issues/homepage URLs;
- PyPI/TestPyPI project/environment configuration for Trusted Publisher identity.

Until these are supplied, `public_rc_authorized=false` and `publish_workflow_enabled=false`.
