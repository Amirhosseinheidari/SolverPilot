# PyPI account setup

The repository is public. The `main` branch requires the 12 CI checks and a pull request. The `pypi` and `testpypi` GitHub environments require approval by Amirhosseinheidari and only accept the `main` branch. No package has been published by this preparation.

1. Register at https://pypi.org/account/register/ and https://test.pypi.org/account/register/ separately. Verify email and enable two-factor authentication on both accounts. Keep recovery codes private.
2. On each account, open Publishing and add a pending GitHub publisher:

| Field | PyPI | TestPyPI |
| --- | --- | --- |
| Project name | solverpilot | solverpilot |
| Owner | Amirhosseinheidari | Amirhosseinheidari |
| Repository | SolverPilot | SolverPilot |
| Workflow filename | publish.yml | publish.yml |
| Environment | pypi | testpypi |

A pending publisher does not reserve the package name. Neither registry had a visible project at the name during the 2026-09-11 preparation check.

The manual publishing workflow requires full qualification and account configuration before it can publish. It defaults to TestPyPI, accepts only successful qualification of a main commit, verifies artifact hashes and GitHub attestations, and uploads only the wheel and source distribution. No permanent API token is needed.

Before publication, run full qualification on the final main commit with attest_artifacts=true. Publish the exact qualified artifacts to TestPyPI, verify installation in a fresh environment, then publish the same bytes to PyPI. Record registry file hashes and create a matching GitHub prerelease. The candidate version is 0.1.0rc2; successful CI alone does not make it a stable 0.1.0 release.
