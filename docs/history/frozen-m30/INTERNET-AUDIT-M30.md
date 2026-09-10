# Internet / Packaging Audit — M30

Audit date: 2026-09-06.

M30 checked current Python packaging guidance and current optional-solver package releases before freezing metadata.

## Python packaging

Current PyPA guidance uses the `[project]` table for metadata such as readme, dependencies, authors/maintainers, project URLs, Python requirements and license metadata. PEP 639 defines SPDX-style `license` expressions and `license-files` for license texts.

M30 intentionally leaves the license, owner identity and project URLs unset because those are project-owner/legal decisions. It also leaves the final public distribution name for owner confirmation.

## Optional solver versions

The evidence-bounded optional dependency ranges remain:

- highspy `>=1.15.1,<2`; PyPI lists highspy 1.15.1 as the July 2, 2026 release.
- osqp `>=1.1.3,<2`; PyPI lists OSQP 1.1.3 as released June 12, 2026.
- pyscipopt `>=6.2.1,<7`; PyPI lists PySCIPOpt 6.2.1 as released May 16, 2026.
- nlopt `>=2.11,<3`; PyPI lists NLopt 2.11.0 as released July 17, 2026.
- CasADi remains pinned to `3.7.2` for the verification extra even though 3.8.0 was released August 25, 2026, because the existing bridge evidence is tied to 3.7.2 and M30 does not transfer that evidence silently.

## Compatibility boundary

The M30 active verification environment is Linux/Python 3.13. Declaring `requires-python >=3.10` is package metadata, not evidence that every Python 3.10–3.13 and OS combination was executed in M30. M31 must run the compatibility matrix before public 1.0 support claims are finalized.
