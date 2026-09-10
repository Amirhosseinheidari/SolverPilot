# SolverPilot Pre-GitHub Cleanup — 0.0.36rc7

rc7 is a release-engineering cleanup of the rc6 SolverPilot brand migration. It introduces no solver algorithm, planner policy, or public API expansion.

## Fixed in rc7

- removed the pre-public `optimind` compatibility package from the distributable source tree; no public release ever established that namespace as a compatibility obligation;
- archived the rc6 compatibility-shim design under `docs/history/frozen-rc6/` for provenance;
- removed the internal `OptiMindError` compatibility alias from live source while preserving historical API snapshots;
- made current release validation derive name/version/Python floor from `pyproject.toml` instead of hard-coding a prerelease number;
- made the distribution manifest derive `PUBLIC-API-RC<n>.json` and `BACKEND-CONTRACT-RC<n>.json` from the current project version;
- made release smoke fail if a legacy `optimind` namespace is present in the installed artifact;
- changed the release-qualification job label to version-neutral wording;
- moved M31/M32/M33 helper scripts from live `tools/` to `docs/history/frozen-tools/`;
- refreshed current pre-GitHub verification/hardening documentation and changelog state.

## Compatibility decision

Because SolverPilot has not had a public 1.0 or public package release under the `optimind` namespace, rc7 chooses a single canonical namespace: `solverpilot`. Historical research artifacts may continue to contain the OptiMind codename, but the public wheel/sdist must not ship an `optimind` package.

## Gate boundary

rc7 may be used as the source baseline for GitHub staging only after its own full regression, distribution build, isolated-install smoke, and archive-integrity checks pass. Public PyPI/1.0 publication remains blocked on external Linux/macOS/Windows qualification and repository/PyPI security configuration.
