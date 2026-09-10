# Security Policy

## Supported version

Before 1.0, only the latest prerelease branch is considered for security fixes. Historical milestone snapshots are retained for provenance but are not maintained release branches.

## Reporting a vulnerability

Do **not** open a public issue for a suspected vulnerability.

Before this repository is made public, the repository owner must enable **GitHub Private Vulnerability Reporting**. Once enabled, use the repository's **Security → Report a vulnerability** flow so the report stays private.

This prerelease repository intentionally does not publish a personal security email address or promise a vulnerability-response SLA that has not been staffed by the project owner.

## Scope

Security reports may include, for example:

- unsafe archive/dataset extraction;
- command or subprocess injection;
- malicious model/input handling that escapes documented resource boundaries;
- dependency or release-workflow compromise;
- credential/permission exposure in CI or publishing workflows;
- artifact-integrity or provenance bypasses.

Solver correctness or performance bugs without a security impact should use the normal issue tracker once the public repository exists.

## Release security requirements

Public release remains blocked until repository security settings are configured, including branch/ruleset protection for the release branch, required CI checks, reviewed GitHub Actions policy, and the protected `pypi` environment/Trusted Publisher configuration described in `RELEASING.md`.

## Trusted benchmark cache boundary

Several research-only scripts under `benchmarks/` use Python `pickle` files for locally
prepared problem caches.  Python pickle is executable serialization and **must never be
loaded from an untrusted download, pull request artifact, shared cache, email attachment,
or other source that is not fully trusted as code**.  These cache files are not a public
data interchange format and are excluded from the installed SolverPilot package.

Official/public benchmark acquisition uses text/archive inputs plus SHA-256/provenance
checks instead.  A future public cache format should use a non-executable canonical
serialization before these research caches are promoted beyond trusted local workflows.
