# PyPI account setup

The repository is public and SolverPilot has an existing PyPI/TestPyPI project.
The current release version is 0.4.1. Ordinary CI contains 22 checks; the attested
release qualification expands to 105 jobs. Inspect live branch/environment
protection before publication; historical setup observations are not proof of
current external settings. Both publication environments are intended to require
owner approval and accept only `main`.

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

Before publication, run full qualification on the final main commit with
`attest_artifacts=true`. Publish the exact qualified artifacts to TestPyPI, verify
installation in a fresh environment, then publish the same bytes to PyPI. Record
registry file hashes, test both published wheel and sdist installations, and create
the matching `v0.4.1` GitHub release. Successful ordinary CI alone is not release
qualification. Historical prerelease records remain unchanged.
