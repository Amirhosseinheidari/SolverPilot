# M30 packaging and dependency audit

This audit records the release-engineering decisions used by M30. It is not a claim that every optional dependency combination was executed in this environment.

## Packaging metadata

Modern Python packaging uses the `[project]` table in `pyproject.toml` for metadata such as project name, version, readme, Python requirement, dependencies, authors/maintainers, license metadata and project URLs. PEP 639 defines SPDX-style `license` metadata and `license-files` for license texts.

M30 intentionally leaves license, maintainer/author and project URLs unset because those values require project-owner decisions. The final distribution name `optimind-core-codename` is also explicitly marked for owner confirmation before public 1.0 publication.

## Runtime dependencies

The base runtime remains deliberately small:

- `numpy>=1.24`
- `scipy>=1.10`

Optional native/public integrations are bounded to the solver major versions for which this project line has evidence:

- HiGHS: `highspy>=1.15.1,<2`
- OSQP: `osqp>=1.1.3,<2`
- SCIP: `pyscipopt>=6.2.1,<7`
- NLopt: `nlopt>=2.11,<3`

The CasADi verification extra remains pinned to `casadi==3.7.2`. CasADi 3.8.0 exists, but the project evidence for the verification bridge was produced on 3.7.2; M30 therefore does not silently transfer that evidence to a newer release.

## Claim boundary

Dependency metadata means an integration is installable/eligible under the declared version range; it does not mean every platform/wheel combination was exercised in M30. Public 1.0 compatibility claims require the M31 CI matrix described in `M30-PUBLICATION-BLOCKERS.md`.
