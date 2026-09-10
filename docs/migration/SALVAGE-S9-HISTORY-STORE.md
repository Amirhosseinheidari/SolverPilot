# SALVAGE-S9 — Optional History / Learning Store

S9 salvages the useful idea behind the legacy OptiMind learning store without
porting its product-specific schemas or automatic workflow persistence.

## Contract

- `solverpilot.history.HistoryStore` is **explicit opt-in**. SolverPilot does not
  create or write a database merely because `solve()` or benchmarking is used.
- SQLite is the only storage dependency and comes from the Python standard library.
- Raw mathematical arrays, raw solution vectors, and raw request payloads are not
  part of the durable schema.
- Durable identities are content-addressed where available (`problem_data_hash`,
  `problem_structural_hash`, versioned `FeatureRecord.record_id`, benchmark `run_id`).
- Existing SolverPilot contracts remain authoritative: `SolveResult`, `SolveTrace`,
  `FeatureRecord`, and benchmark JSONL rows are adapted; parallel problem/benchmark
  models are not introduced.
- Writes use explicit transactions with rollback; foreign keys are enabled.
- Schema evolution is stepwise through SQLite `PRAGMA user_version`; databases newer
  than the installed code fail closed.

## Tables

- `problems`: hashes, problem class, dimensions, optional source provenance metadata.
- `feature_records`: versioned pre-solve feature records from S4.
- `solve_runs`: compact facts from `SolveResult`; no `x` and no raw statistics.
- `benchmark_rows`: normalized identity/timing columns plus a compact row copy with
  opaque trace/solution payloads stripped.
- `artifacts`: metadata only (ID/type/path/size/hash/metadata).

## Privacy / scope

This is a local research-history foundation, not a customer-history service. It
makes no tenant-isolation, retention-policy, encryption-at-rest, remote-service, or
multi-user production-persistence claim. Paths and metadata supplied explicitly by
callers are stored as provided; callers should avoid secrets and personal data.

## Idempotency and immutability

Canonical IDs are append/idempotent. Re-recording identical content is safe;
reusing the same problem hash, feature `record_id`, solve `run_id`, benchmark
`run_id`, or artifact ID with different durable content fails closed instead of
overwriting history. Multiple source-provenance records may point to the same
problem data hash through the separate `problem_sources` table.
