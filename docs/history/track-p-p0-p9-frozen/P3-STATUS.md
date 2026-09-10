# P3 STATUS — Backend Capability Protocol 2.0

**Package version:** 0.0.34  
**Track:** P — Platform / Compiler Architecture  
**Status:** VERIFIED RELEASE

## What P3 adds

P3 replaces the coarse capability vocabulary as the *future compiler contract* with a granular, version-bound, fail-closed capability protocol while keeping the legacy `BackendManifest` API intact for Track-M compatibility.

Implemented:

- `BackendCapabilityManifestV2`
- granular `CapabilityKey` namespace across problem classes, constraints, incremental edits, starts, callbacks, result artifacts, and lifecycle
- `CapabilityClaim` with support state, implementation mode, restrictions, verification state, and evidence
- exact runtime identity binding across backend / binding / OptiMind adapter versions
- deterministic capability signatures for compiler/cache identity
- deep immutability of claims/restrictions/metadata
- conservative legacy-manifest projection that **never upgrades old declarations to verified v2 evidence**
- v2 problem requirements and fail-closed compatibility checks
- native/adapter conformance kit for currently exercisable solve/result channels
- typed future backend protocol for rich artifacts, starts, callbacks and incremental operations
- `optimind-capabilities` CLI
- packaged P0-compatible capability JSON schema
- registry helpers for projected or verified v2 manifests

## Critical trust rules

1. `native` in the legacy manifest does not mean `VERIFIED` in v2.
2. Upstream solver documentation does not by itself create an OptiMind adapter capability claim.
3. Evidence is valid only for the recorded backend/binding/adapter runtime identity.
4. Safe or risky emulation is never silently treated as native support.
5. Coarse legacy same-sparsity/structural update declarations are projected as restricted granular claims until each operation has dedicated conformance.
6. A solver termination does not imply primal/dual/basis/IIS/ray availability.
7. Capability manifests are deeply immutable so a capability signature cannot drift after construction.

## Actual runtime conformance inventory

The P3 matrix contains 13 built-in backend candidates in this environment:

- 10 available and conformance-smoke passing
- 3 optional public native adapters unavailable (`highspy-native`, `osqp-native`, `pyscipopt-native`)

Verified channels are intentionally narrow: problem-class solve support, linear-constraint path, primal result channel, and objective result channel where exercised by the independent health smoke.

Advanced historical declarations such as basis starts, same-sparsity updates, infeasibility certificates, or reoptimization are **not promoted to granular VERIFIED v2 claims** merely because the old manifest or upstream solver API says they exist. Their dedicated adapter/conformance work belongs to later Track-P milestones.

## Regression

```text
323 collected
320 passed
3 skipped
0 failed
0 errors
```

The three skips remain the same optional public native-package integration tests.

## Scope intentionally deferred

P3 does not implement:

- the P4 Bridge/Reformulation Engine;
- backend-bound compilation or effective bridged capability;
- the P5 persistent backend session;
- production callback handlers;
- granular native conformance for every advanced capability;
- solver selection or performance routing.

The existing P2 compiler remains backend-neutral. P4 will consume P3 verified manifests and write the selected capability signature/transformation path into compilation identity.

## Wheel verification

Wheel: `dist/optimind_core_codename-0.0.34-py3-none-any.whl`  
SHA-256: `c38cf1102a0b74914e7bb4b54e532e066c6fea0632b004e606278ab2dde8e56d`

Out-of-source wheel verification confirmed:

- package/distribution version `0.0.34`;
- source-tree import = false;
- packaged backend capability v2 schema;
- verified SciPy/HiGHS LP conformance;
- legacy projection remains unverified;
- exact version-drift rejection;
- P2 miss → partial-recompile → snapshot-hit behavior;
- legacy direct LP solve and independent validation;
- `optimind-capabilities --verify`;
- full 13-backend wheel capability matrix matches the source matrix signatures.
