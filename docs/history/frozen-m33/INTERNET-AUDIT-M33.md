# Internet / Standards Audit M33 — 2026-09-06

Primary references were rechecked before M33 implementation.

## Scientific Python support policy

Scientific Python SPEC 0 recommends dropping Python versions three years after initial release and core package dependencies two years after initial release. Its current support table places Python 3.11 outside the recommended window, Python 3.12 inside the window until Q4 2026, NumPy 2.2 inside the window until December 2026, and SciPy 1.15 inside the window until January 2027.

Source: https://scientific-python.org/specs/spec-0000/

Python 3.10's PEP 619 lifecycle provides source-only security fixes until approximately October 2026.

Source: https://peps.python.org/pep-0619/

Current NumPy 2.5.2 and SciPy 1.18.1 both require Python >=3.12 and publish wheels for current interpreters including 3.14.

Sources:
- https://pypi.org/project/numpy/2.5.2/
- https://pypi.org/project/scipy/1.18.1/

Current optional solver releases Highs 1.15.1, OSQP 1.1.3, PySCIPOpt 6.2.1, NLopt 2.11.0 and pinned CasADi 3.7.2 expose CPython 3.14 wheels on major desktop platforms. This is availability evidence only.

Sources:
- https://pypi.org/project/highspy/
- https://pypi.org/project/osqp/1.1.3/
- https://pypi.org/project/PySCIPOpt/6.2.1/
- https://pypi.org/project/nlopt/2.11.0/
- https://pypi.org/project/casadi/3.7.2/

## Secure release process

Scientific Python SPEC 8 recommends documenting the release process, minimal default workflow permissions, protected release environments, full-SHA Action pinning, Dependabot-reviewed Action updates, artifact attestations and OIDC Trusted Publishing.

Source: https://scientific-python.org/specs/spec-0008/

GitHub states that pinning an Action to a full-length commit SHA is the only immutable Action reference. GitHub artifact attestations bind artifact digest to workflow/repository/commit provenance and use Sigstore.

Sources:
- https://docs.github.com/en/actions/reference/security/secure-use
- https://docs.github.com/en/actions/concepts/security/artifact-attestations

PyPI Trusted Publishing exchanges short-lived OIDC identity for publication credentials, avoiding long-lived API tokens. PyPI's PEP 740 implementation supports publish and SLSA provenance attestations, and the official PyPA publishing action produces upload attestations automatically by default when publishing through Trusted Publishing.

Sources:
- https://docs.pypi.org/trusted-publishers/
- https://docs.pypi.org/attestations/
- https://docs.pypi.org/attestations/producing-attestations/
- https://peps.python.org/pep-0740/

## Runtime vulnerability audit

PyPA `pip-audit` audits Python environments/requirements against vulnerability services and can emit CycloneDX JSON/XML SBOMs. M33 pins audit tooling to `pip-audit==2.10.1` in an isolated audit venv.

Source: https://github.com/pypa/pip-audit

## Publishing action template

The current `pypa/gh-action-pypi-publish` latest release is v1.14.2 at commit `dc37677b2e1c63e2034f94d8a5b11f265b73ba33`. M33's publication template is disabled and uses this full SHA.

Source: https://github.com/pypa/gh-action-pypi-publish/releases
