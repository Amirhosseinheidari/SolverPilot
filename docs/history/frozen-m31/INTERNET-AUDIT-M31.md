# M31 Internet / Standards Audit — 2026-09-06

M31 rechecked current packaging and CI conventions because the release-candidate phase is sensitive to metadata and platform claims.

## PyPA metadata

Current Python Packaging User Guide/Core Metadata documentation uses SPDX `License-Expression` semantics (PEP 639) and `license-files`; authors/maintainers and project URLs are independent project metadata. The current PyPA guide notes PEP-639 support in setuptools beginning at 77.0.3, so M31 raises the build-system floor to `setuptools>=77.0.3` while deliberately leaving license and owner identity fields unset until the project owner chooses them.

Sources:
- https://packaging.python.org/en/latest/guides/writing-pyproject-toml/
- https://packaging.python.org/en/latest/specifications/core-metadata/
- https://packaging.python.org/en/latest/specifications/pyproject-toml/

## GitHub Actions

GitHub recommends explicit `actions/setup-python` rather than relying on runner-default Python. The M31 workflow uses explicit Python 3.10-3.13 and Linux/macOS/Windows matrices with `actions/checkout@v6` and `actions/setup-python@v6`.

Sources:
- https://docs.github.com/en/actions/tutorials/build-and-test-code/python
- https://github.com/actions/setup-python
- https://github.com/actions/checkout

## Optional dependency artifact availability

PyPI was checked for the project-pinned/bounded optional integrations. Current releases expose binary wheels spanning CPython 3.10-3.13 on the major desktop OS families for Highs (`highspy 1.15.1`), OSQP (`1.1.3`), PySCIPOpt (`6.2.1`), NLopt (`2.11.0`) and pinned CasADi (`3.7.2`). This is recorded only as upstream artifact availability, never as OptiMind execution evidence.

Sources:
- https://pypi.org/project/highspy/
- https://pypi.org/project/osqp/1.1.3/
- https://pypi.org/project/PySCIPOpt/6.2.1/
- https://pypi.org/project/nlopt/2.11.0/
- https://pypi.org/project/casadi/3.7.2/

## Naming / branding collision

A Microsoft/OptiGuide research project publicly uses the name `OptiMind` for optimization-reasoning research. M31 treats this as a branding review blocker, not as a legal conclusion. The project owner must decide the public name before 1.0.

Source:
- https://github.com/microsoft/OptiGuide
