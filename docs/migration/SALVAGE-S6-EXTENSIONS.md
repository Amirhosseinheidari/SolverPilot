# SALVAGE-S6 — Extension Architecture

Status: migration-stage architecture hardening; not a public/PyPI release.

## Scope

S6 introduces `solverpilot.extensions` as an explicit manager for trusted, in-process Python extensions. It does **not** replace SolverPilot's canonical `BackendRegistry` or `BridgeRegistry`, does not auto-discover third-party code on import, and does not claim sandbox isolation.

Supported extension contributions are staged before activation. Backend and bridge contributions are ultimately applied to caller-supplied canonical registries; connector, feature, reporter, evaluator, and application contributions are retained as named exports after successful activation.

## Safety and reproducibility invariants

- extension IDs are normalized, unique, and frozen before activation planning;
- dependency requirements use deterministic, case-preserving SemVer checks and fail on malformed, missing, incompatible, or cyclic dependencies;
- manifests are snapshotted at registration and drift is rejected;
- self-check is repeated before staging;
- contribution hooks execute outside the manager's central lock;
- activation uses a manager-issued, single-use plan;
- forged, stale, or mutated plans are rejected;
- staged backend observable state (identity, manifest name/version/capabilities/metadata) is fingerprinted; non-finite/cyclic state and post-stage drift fail closed;
- activation plans are bound to the exact backend/bridge registry objects used at planning time, and target registry state drift invalidates the plan;
- canonical registry collision rules are previewed before mutation;
- non-`mixed` manifest kinds may only contribute their declared extension category;
- extension activation does not create parallel backend or bridge registries;
- importing `solverpilot.extensions` does not resolve or execute third-party entry points.

## Trust boundary

Extensions are trusted Python code running in the current process. S6 provides validation, deterministic dependency ordering, explicit staging, and reproducibility guards; it is not a security sandbox. Callers must not treat arbitrary third-party extensions as isolated from the host process.

## Public API policy

S6 is exposed from the `solverpilot.extensions` subpackage only. The frozen top-level `solverpilot.__all__` surface remains unchanged.
