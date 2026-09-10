# Contributing

SolverPilot is still pre-1.0. Changes should preserve the project's fail-closed evidence standard: do not promote an unexecuted compatibility cell, skipped optional integration, or historical benchmark result into a current pass.

## Development setup

```bash
python -m pip install ".[test]"
python -m pytest
```

For benchmark work:

```bash
python -m pip install ".[benchmark]"
```

For optional public solver integrations, install the relevant extra (`highs`, `osqp`, `scip`, `nlopt`, or `casadi`).

## Pull requests

A pull request should:

- include regression tests for behavioral changes;
- keep public-status and evidence claims scoped to what the tests actually establish;
- preserve deterministic/provenance fields in benchmark artifacts;
- avoid adding generated wheels, sdists, caches, benchmark corpora, or credentials to the repository;
- update public API/release documentation when externally visible behavior changes.

Normal PR CI is intentionally smaller than release qualification. Cross-platform release claims come only from the manual exact-artifact qualification workflow.

## API changes

The top-level public API was frozen before 1.0. Backward-compatible additions require explicit review; removals or semantic changes should not be slipped into cleanup work. `solverpilot.experimental` is not covered by the same compatibility promise.

## Security

Follow `SECURITY.md`. Do not disclose suspected vulnerabilities in public issues or pull requests.

## Contribution licensing

Unless explicitly stated otherwise, contributions intentionally submitted for inclusion in this project are provided under the Apache License 2.0, consistent with Section 5 of the license.
