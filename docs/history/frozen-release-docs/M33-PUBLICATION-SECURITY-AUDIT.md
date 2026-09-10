# M33 Publication and Supply-Chain Security Audit

## Support-window decision

M33 narrows the intended 1.0 interpreter floor from Python 3.10 to Python 3.12 and adds Python 3.14. The intended core matrix is therefore Python 3.12–3.14 across Linux, macOS and Windows.

This is based on current ecosystem support rather than a failed-test workaround: M32 had no failed core cells; all were unexecuted. Python 3.10 is near upstream security EOL in October 2026, Scientific Python SPEC 0 recommends a three-year Python support window, and current NumPy/SciPy releases require Python >=3.12. The principal optional solver bindings used by OptiMind publish 3.14 wheels on the major desktop OS families.

M33 also raises the runtime dependency floors to NumPy >=2.2 and SciPy >=1.15 and adds a minimum-dependency regression job.

## Release-process security

The release process follows the direction of Scientific Python SPEC 8 and current GitHub/PyPI guidance:

- default workflow permission is `contents: read`;
- Actions are pinned to full commit SHA;
- Dependabot is configured to propose reviewed GitHub Action SHA updates;
- build and publication privileges are separated;
- artifact provenance attestations are generated and verified;
- publication, when eventually enabled, uses a protected GitHub environment and PyPI Trusted Publishing/OIDC;
- long-lived PyPI API tokens are not part of the intended path.

A disabled publication template is checked in as `.github/workflows/m33-publish.yml.disabled`; it is intentionally not executable by GitHub Actions until owner metadata and all technical gates pass.

## Runtime vulnerability / SBOM gate

Each external core qualification cell creates an isolated runtime environment, records exact runtime versions, passes `pip check`, and writes exact runtime dependency pins. A separate audit environment installs `pip-audit==2.10.1`, audits those runtime requirements, and retains CycloneDX JSON output. No vulnerability-ignore list is pre-approved.

This separation prevents the audit tool's own dependencies from being mislabeled as OptiMind runtime dependencies.

## External state that source code cannot prove

Public RC requires repository-state verification for:

- branch protection;
- release environment required reviewers;
- allowed-actions policy;
- final Trusted Publisher identity;
- owner/legal metadata and branding decisions.

These remain fail-closed until checked against the actual publication repository.
