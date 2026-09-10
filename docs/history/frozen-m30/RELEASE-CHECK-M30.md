# Release Check — M30

## Software gates

- Version coherence (`0.0.35`): PASS
- Stable public API manifest (78 symbols): PASS
- Research API separated under `optimind.experimental`: PASS
- Stable public exception hierarchy: PASS
- Backend contract covers built-in candidates: PASS
- Learned LP performance auto-routing frozen off: PASS
- Full regression: PASS (280 passed / 3 skipped / 0 failed / 0 errors)
- Property/numerical campaigns: PASS
- Final wheel build: PASS
- Final wheel integrity: PASS
- Final wheel isolated-target solve/CLI smoke: PASS
- Final sdist build: PASS
- Final sdist round-trip wheel build/install/solve: PASS
- Source provenance reconciliation: PASS
- Final source manifest: PASS
- Final internal SHA-256 inventory: PASS
- Final M30 ZIP integrity and member-level rehash: PASS
- Historical M1–M30 benchmark archive integrity: PASS

## Distribution metadata observations

The final M30 wheel intentionally has no License-Expression, author/maintainer metadata, or Project-URL fields. These are unresolved owner/publication inputs, not inferred values.

The distribution name remains `optimind-core-codename`; M30 does not assume that this is the owner's final public project name.

`requires-python >=3.10` is a compatibility declaration, not proof that every advertised interpreter/OS combination has been exercised. The final M30 runtime verification environment is Linux / Python 3.13; M31 owns the public compatibility matrix.

## M30 gate

**PASS — FINAL VERIFIED M30 RELEASE-CONSOLIDATION.**

## Public 1.0 gate

**NOT YET PASSED.** M31 still requires owner metadata/license decisions and the intended Python/OS compatibility matrix. M30 is the frozen release-consolidation baseline for that phase.
