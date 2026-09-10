# M30 release policy

M30 is the consolidation checkpoint immediately before the release-candidate phase.

## Hard gates

A release artifact must satisfy all of the following:

1. `optimind.__version__` and `pyproject.toml` version agree.
2. Public API manifest exactly matches `optimind.__all__`.
3. Research-only milestone APIs are absent from the frozen public list.
4. Full regression has zero failures/errors; optional dependency skips are counted, not called passes.
5. Wheel builds without network dependency resolution and installs outside the source tree.
6. Installed-wheel smoke covers LP, MILP, QP, validation, conservative production planning, public exceptions, and CLI entry points.
7. Source/wheel contain no generated `__pycache__`, `.pyc`, source-tree `egg-info`, or stale build directories.
8. Release ZIP and benchmark archive have internally verified SHA-256 manifests and valid ZIP structure.
9. No learned LP performance selector is enabled without valid held-out evidence.
10. Historical prototype evidence is not represented as source functionality when the implementation bytes are absent.

## Publication blockers reserved for M31

The project owner must choose/provide:

- final public distribution/project name if `optimind-core-codename` is not intended to ship,
- license/SPDX expression and license text,
- author/maintainer metadata,
- source/documentation/issue-tracker URLs.
- Python/OS compatibility matrix for every platform/version claimed by public 1.0.

M30 does not invent these values.
