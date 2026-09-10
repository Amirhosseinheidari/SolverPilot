# M30 publication and compatibility blockers

M30 closes the software consolidation gates without inventing project-owner metadata. Public 1.0 publication remains blocked until the following governance and compatibility decisions are completed.

## Owner/governance decisions

1. **License** — choose an SPDX license expression and provide the matching license text/file(s). Do not infer a license from dependencies or repository history.
2. **Distribution/project name** — confirm whether `optimind-core-codename` is the final public distribution name.
3. **Author/maintainer identity** — provide names/emails if they should appear in package metadata.
4. **Project URLs** — provide/confirm source repository, documentation, issue tracker and/or homepage URLs.

These are identity/legal decisions and M30 deliberately leaves them unset rather than fabricating values.

## Compatibility verification required before public 1.0 claims

The M30 release candidate is exercised in the active Linux/Python 3.13 runtime. Although `pyproject.toml` currently declares `requires-python = ">=3.10"`, M30 does **not** claim that Python 3.10, 3.11 or 3.12, Windows, macOS, alternative architectures, or every optional native dependency combination were executed in the M30 verification environment.

Before a public 1.0 release, M31 should run a clean CI compatibility matrix covering the Python versions and operating systems the project intends to advertise. Unsupported combinations should either be fixed or removed from the published compatibility claim.
