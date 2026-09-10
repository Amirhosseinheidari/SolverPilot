# SALVAGE-S10 — Cross-system hardening and release integration

S10 adds no new optimization family. It treats the S9 verified source as the authoritative baseline and audits the S2–S9 additions as one system.

## Hardening fixes made in S10

1. **Oracle evidence classification** — `meets_or_beats_best_known` is no longer treated as exact-reference evidence. Exact-reference eligibility now accepts `objective_matches_optimum`; reference-free `valid_optimal` rows remain solver-optimal candidates, not an independent global proof.
2. **Oracle comparability identity** — when protocol/environment/instance SHA comparison is required, missing/blank identities now fail closed; instance SHA must be a valid 64-hex digest.
3. **Oracle policy typing** — boolean policy flags reject truthy strings and other non-bool values.
4. **ExplanationReport immutability** — structured evidence mappings are deep immutable snapshots. Mutating caller-owned dictionaries after report construction cannot drift the report.
5. **Explanation claim type boundary** — claim kind/disposition are normalized to enums, closing a manual-constructor bypass where strings could evade identity-based claim checks.
6. **Explanation render boundary** — JSON/Markdown rendering refuses a report that no longer satisfies claim-safety invariants.
7. **Application proof issuance** — public `TSPSolution` / `VRPSolution` constructors cannot self-issue `optimality_proven=True`; SolverPilot verified/reference paths carry an internal proof token.
8. **Cross-system integration test** — strict JSON provenance → canonical problem → feature record → solve → explanation → history → benchmark oracle/regret is exercised end to end.

## Remaining limitations / weaknesses

### High priority

- **External concurrent registry mutation:** `ExtensionManager` serializes its own activation operations and prevalidates targets, but canonical backend/bridge registries are caller-owned and are not globally transactional against unrelated direct concurrent mutation. Such mutation violates the documented activation precondition and can leave a target requiring inspection after a failed commit.
- **Unified intelligence/history coverage:** feature extraction and automatic history/problem adapters are still centered on canonical LP/MILP/QP. Extended Track P families and TSP/VRP do not yet feed a single generic feature/history/report schema.

### Medium priority

- **OOD is univariate and uncalibrated:** range, z-score, missingness and unseen-category guards are interpretable safety signals, not a probability that a selector will fail.
- **Filesystem race boundary:** strict I/O rejects static symlinks/path escape and limits read size, but it is not a hardened filesystem sandbox against malicious concurrent path replacement between validation and open.
- **History confidentiality:** SQLite history is plaintext and unencrypted. Raw arrays/vectors are excluded by default, but explicit caller metadata/path fields may still contain sensitive information.
- **Reference scalability:** TSP/VRP exhaustive references are exponential; MILP formulations are correctness references for small/medium workloads, not a claim of best-in-class routing scale.
- **History migration maturity:** schema migration infrastructure exists, but the current durable schema is only v1; a real historical v1→v2 production migration has not yet been exercised.

### Release/environment qualification

- Exact public-release build pins (`setuptools==84.0.0`, `wheel==0.48.0`) must still pass in the external release workflow.
- OR-Tools CP-SAT optional integration must be re-qualified in an environment where the pinned OR-Tools runtime is installed.
- Package version remains `0.0.40rc2` inside migration artifacts; a public release must choose and verify a new release version deliberately.

## Performance guard

A repeated clean-interpreter probe compares S9 and S10. The recorded median ratios are evidence of no material regression in the probe workload, not a general performance theorem.
