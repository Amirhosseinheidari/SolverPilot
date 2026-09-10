# M32 Internet / Standards Audit — 2026-09-06

M32 re-audits external standards because compatibility and publication claims are now the primary release risk.

## Python packaging / PEP 639

Current PyPA guidance uses an SPDX license expression plus `license-files` for PEP-639-style licensing metadata. Authors/maintainers and project URLs are separate metadata. Setuptools support for this declaration is documented beginning at 77.0.3; the project already uses `setuptools>=77.0.3` as its build-system floor.

Primary references:
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://packaging.python.org/en/latest/specifications/core-metadata/
- https://packaging.python.org/en/latest/specifications/pyproject-toml/

M32 does not invent license, owner or project URLs.

## Python / NumPy / SciPy support drift

The declared package floor is Python >=3.10 with `numpy>=1.24` and `scipy>=1.10`. This range allows pip to select interpreter-compatible releases, but the latest scientific-stack releases no longer cover every declared Python version.

Examples observed during the M32 audit:
- current NumPy 2.5.2 requires Python >=3.12;
- current SciPy 1.18.1 requires Python >=3.12;
- SciPy 1.17.1 requires Python >=3.11;
- SciPy 1.15.3 supports Python >=3.10;
- NumPy 2.2.x supports Python >=3.10.

Therefore M32 records the actual NumPy/SciPy resolver result per matrix cell and requires `pip check`; it never equates “latest dependencies” with Python-3.10 compatibility.

Primary references:
- https://pypi.org/project/numpy/
- https://pypi.org/project/scipy/
- https://pypi.org/project/scipy/1.15.3/

## GitHub Actions security and compatibility

GitHub recommends explicit `actions/setup-python` for Python workflows and documents matrix testing across hosted OS runners. GitHub's secure-use reference says pinning to a full-length commit SHA is the only immutable reference for an action.

Primary references:
- https://docs.github.com/en/actions/tutorials/build-and-test-code/python
- https://docs.github.com/en/actions/how-tos/security-for-github-actions/security-guides/security-hardening-for-github-actions
- https://github.com/actions/setup-python

M32 pins every external action in its release-compatibility workflow to a 40-character SHA.

## Build provenance and Trusted Publishing

GitHub artifact attestations bind an artifact digest to workflow identity/provenance. PyPI Trusted Publishing uses short-lived OIDC credentials instead of manually managed long-lived tokens. PyPI digital attestations bind uploaded release files to publisher identity/workflow; the official PyPA publishing action can generate/upload attestations automatically when used with Trusted Publishing.

Primary references:
- https://docs.github.com/en/actions/concepts/security/artifact-attestations
- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/attestations/
- https://github.com/pypa/gh-action-pypi-publish

M32 prepares these controls but keeps publication disabled pending owner metadata.

## Exact optional-solver artifact availability

PyPI file metadata confirms CPython-3.13 Linux x86-64 wheels for:

- highspy 1.15.1 — SHA256 `238b2ee88b974b21c7e9ef198139502a7d87451939cae143dce789bbda121182`;
- OSQP 1.1.3 — SHA256 `2d3ee63e8c65ef89fce979c068d05bfc3ed92b1bbc4246fbacc663f86cbe02b2`;
- PySCIPOpt 6.2.1 — SHA256 `725503ea90fd0962f1111a5b46ee3e271b7375e1b0dfda708ce223dbebaeff5e`.

Equivalent CPython 3.10-3.13 and Windows/macOS wheel families are also published. This is **availability evidence only**.

The M32 container attempted to download the exact Linux wheels from `files.pythonhosted.org`, but binary download/network access failed; no native-backend execution pass is inferred from this.

Primary references:
- https://pypi.org/project/highspy/1.15.1/#files
- https://pypi.org/project/osqp/1.1.3/#files
- https://pypi.org/project/PySCIPOpt/6.2.1/#files

## Branding review

Microsoft's OptiGuide repository publicly contains an `optimind` research project/paper. M32 treats this as a naming-review flag, not a legal conclusion. The owner must explicitly choose the final public distribution/brand before publication.

Reference:
- https://github.com/microsoft/OptiGuide
