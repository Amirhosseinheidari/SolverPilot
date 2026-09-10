# OptiMind M22 — Official Corpus Finalization Gate

**Result:** PASS

## Official corpus integrity

- MIPLIB 2017 benchmark-v2: reconstructed from 2 Drive chunks; 332,664,623 bytes; SHA-256 `c756eefd544d83b31809306b45d3549a1a5b9378e6aa78b68738b1a3b6a418fa`; ZIP integrity PASS; 240 entries.
- QPLIB archive: reconstructed from 4 Drive chunks; 774,972,035 bytes; SHA-256 `b3596e1264ed57c5f6a44e822679f5c9138e1985fe74bc8341c3becbc666b9fd`; ZIP integrity PASS; 4,069 archive entries / 453 `.qplib` files. M22 executed the 19 predeclared current-IR candidates.
- PACE 2019 Vertex Cover Exact public: 10,695,845 bytes; SHA-1 `e7ca305528a0257235a95c41742f2b3431e1e485` matches the official published SHA-1; SHA-256 `c78177cb5b2f7468bc652bd6e082f002495b26ba220bdc8eee9993ac6ff834c6`; exactly 100 graph files.

## Repeated official-byte execution

All campaigns were executed on one host with 12 controller workers, solver thread count 1, and three repetitions per instance. Solver/controller timeouts are explicit outcomes; missing outcomes are forbidden.

- MIPLIB: 240 unique instances × 3 = **720/720 outcomes**; 0 read/worker errors; 0 reference mismatches; 324 hard controller timeouts and 396 solver time-limit outcomes under the deliberately short smoke budget.
- QPLIB: 19 unique candidates × 3 = **57/57 outcomes**; 0 read/worker errors; 0 reference mismatches; 27 hard controller timeouts and 30 solver time-limit outcomes.
- PACE: 100 public instances × 3 = **300/300 outcomes**; 0 read/worker errors; **81 real incumbents independently validated against original graph edges**, 0 invalid solutions; 32 hard controller timeouts and 268 solver time-limit outcomes.

## Extra reference cross-checks

- MIPLIB `neos5`: OptiMind bundled HiGHS returned a feasible-limit candidate with objective `14.999999999999972`; independent canonical validation PASS; official optimum reference = `15.0`; match PASS.
- QPLIB `QPLIB_8792`: HiGHS returned `optimal`, objective `3593.5162939294355`; QPLIB best-known reference = `3593.518355`; meets/beats best-known PASS.

## Claim boundary

This PASS certifies the M22 **official-byte acquisition/integrity/outcome-accounting/fail-closed validation gate** under the declared timeout-permitted protocol. It does **not** claim that every public instance was solved to optimality, nor that OptiMind is faster than mature solvers. Performance-policy claims still require longer-cutoff benchmark analysis.
