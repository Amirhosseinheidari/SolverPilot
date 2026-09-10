# SALVAGE-S2 — Safe I/O + Provenance

Status: implementation candidate; requires focused, full-regression, and installed-wheel gates.

## Scope

S2 adds an **additive** `solverpilot.io` subpackage. It does not modify SolverPilot's
mathematical problem IR, backend registry, runtime routing, validation semantics, or
frozen top-level `solverpilot.__all__`.

The legacy OptiMind connector foundation was used as a source of tested invariants.
Legacy plugin wrappers, target adapters, TSP/VRP materializers, organizational
metadata, and connector registries were intentionally not ported.

## New architecture

```
local JSON / CSV
      ↓
source safety + bounded read
      ↓
strict parser
      ↓
explicit deterministic mapping
      ↓
immutable normalized payload
      +
content-addressed SourceProvenance
```

`SourceProvenance.as_problem_metadata()` returns a detached metadata fragment that can
be supplied to existing SolverPilot problem constructors without changing their
mathematical structural/data hashes.

## Preserved fail-closed invariants

- duplicate JSON keys rejected
- NaN / Infinity rejected
- configurable strict decimal-to-binary64 policy
- excessive JSON integer digits rejected
- nesting depth bounded
- invalid Unicode rejected
- symlinked source paths rejected by default
- declared CSV bundles only; no directory discovery
- path traversal / drive-component escapes rejected
- bounded reads enforce a total source-size budget
- duplicate/blank CSV headers rejected
- inconsistent/blank CSV rows rejected
- mapping target collisions rejected
- mapping defaults/constants detached and immutable
- provenance hashes normalized and immutable
- errors avoid echoing source values

## Deliberate redesigns

- No legacy `ConnectorRequest` or target-adapter model.
- No automatic TSP/VRP materialization in the I/O layer.
- No plugin dependency from I/O.
- `strict_numbers=True` preserves the exact legacy decimal policy; users may opt into
  finite binary64 rounding with `strict_numbers=False`.
- Provenance is smaller and generic to SolverPilot.
- New symbols live under `solverpilot.io` and are not added to the frozen top-level API.

## Final verification — 2026-09-07

Status: **FINAL VERIFIED MIGRATION STAGE (not a public/PyPI release)**.

Verification evidence:

- RC2 regression: 593 unique tests, 591 passed, 2 known external-data skips, 0 failures/errors.
- S2 focused source tests: 62/62 passed.
- Combined source inventory: 655 unique tests, 653 passed, 2 skipped, 0 failures/errors, 0 duplicate test IDs.
- Batch-1 clean-exit replay: 198/198 original test IDs reproduced exactly, 0 missing/extra/duplicate IDs.
- Installed-wheel S2 tests: 62/62 passed outside the source tree.
- Legacy strict-semantics differential campaign: 4500/4500 comparisons matched.
- Direct wheel smoke: JSON, CSV, provenance/hash invariance, and core LP solve passed.
- sdist round-trip wheel smoke: the same checks passed outside the source tree.
- Direct and sdist-rebuilt wheels contain identical member content excluding RECORD; archive bytes are not claimed identical.
- Frozen top-level public API remains 78 symbols; S2 symbols are available only under `solverpilot.io`.

S2 deliberately does **not** add target materializers, TSP/VRP adapters, plugin registries,
or solver-selection behavior. Those remain later migration stages.
