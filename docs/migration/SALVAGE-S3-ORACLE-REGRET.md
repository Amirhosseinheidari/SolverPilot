# SALVAGE-S3 — Oracle + Regret

Status during implementation: **IN PROGRESS until all verification gates pass**.

## Scope

S3 salvages the scientifically useful oracle/regret semantics from legacy OptiMind without importing its old evaluation data model, plugin layer, gate engine, or evidence bureaucracy.

The authoritative inputs are SolverPilot benchmark rows. Existing `benchmark.summary` and policy aggregation remain unchanged.

## Core invariants

- Oracle construction is objective-direction aware.
- Only solved, independently validated, exact-success rows with finite objective/runtime enter the objective oracle.
- `@auto` rows are excluded from the direct-run oracle by default.
- Protocol, environment, and instance content identity mismatches fail closed by default.
- Objective ties are preserved; runtime tie-breaking is optional and explicit.
- Missing/failed/timeout/limit/infeasible selections never receive fabricated numeric quality regret.
- Quality regret, runtime delta, selection overhead, and binary failure dimensions remain separate.
- Relative quality regret is undefined when the oracle objective is zero.
- Summary statistics preserve missingness and do not make significance claims.

## Deliberately not migrated

- Legacy `BenchmarkObservation`, `OracleEntry`, and selector data model.
- Legacy evaluation gate/release machinery.
- Legacy plugin wrappers and serialization contracts.
- Arbitrary scalarization of quality, runtime, timeout, and failure regret.

## API boundary

New APIs live under `solverpilot.evaluation`; no new top-level `solverpilot.*` symbols are introduced.
