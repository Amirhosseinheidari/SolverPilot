# M28 Native-Choose Baseline Audit Protocol

Status: **frozen before M28 outcome collection**

## Question
Before another learned LP selector is attempted, determine whether HiGHS' own default/`choose` route already closes enough of the DS-vs-IPM opportunity to make an external OptiMind selector unnecessary.

## Important semantics boundary

- SciPy 1.17.0 `linprog(method="highs")` passes `solver=None` to `_linprog_highs`; its own implementation documents that `None` uses simplex. It is therefore a semantics probe, **not** an independent automatic algorithm-selection competitor.
- The current HiGHS documentation (September 2026) states `solver="choose"` selects the default solver for LP, which is simplex. That documentation describes current HiGHS, while M28's timing-comparable native core is the verified bundled HiGHS **1.10.0** shipped with CasADi 3.7.2.
- Performance comparisons must not mix solver-core versions. The primary cost matrix therefore uses only bundled HiGHS 1.10.0 through the same direct C API with `solver in {choose, simplex, ipm}`.

## Fresh public/OOD cohort

Source: official MIPLIB 2017 benchmark-v2 archive already SHA-locked by M22.

Selection is outcome-independent:

1. Start from the official 240-name `benchmark-v2.test` manifest.
2. Exclude every M25 cohort instance (therefore also excluding the M26 split and M27 consumed held-out instances).
3. Keep archive members with compressed size <= 2 MiB.
4. Parse with OptiMind's MPS parser and convert integrality to a continuous LP relaxation.
5. Parse each candidate in a separate process with an 8 s controller hard timeout; parse timeout/error is an explicit outcome-independent exclusion.
6. Keep only models with `n <= 12000`, `m <= 12000`, `nnz <= 300000`.
7. Order eligible names by SHA-256(filename).
8. Take the first 48.

The initial 512 KiB / 5000 / 5000 / 100000 envelope yielded only 20 fresh instances before any M28 solve outcomes existed. A 2 MiB / 10000 / 10000 / 250000 envelope with process-isolated parsing yielded 41. Before any solver timing existed, a structural availability audit showed the smallest tested expansion reaching at least 48 was `n,m <= 12000`, `nnz <= 300000` (51 eligible); M28 therefore froze that minimal expansion and re-hashed the protocol before timing collection.

The cohort file, protocol file, runner files, corpus archive, manifest, and prior-cohort artifact hashes are embedded in raw evidence.

## Execution protocol

- 48 fresh instances.
- 3 rounds per instance.
- Algorithms: `choose`, `simplex`, `ipm` using the same bundled HiGHS 1.10.0 C API.
- Solver threads: 1.
- Common C/BLAS thread environment: 1.
- Solver time limit: 1.0 s.
- Controller hard timeout: 3.0 s.
- Process-isolated solve workers.
- Deterministic interleaving of algorithm order by `(instance_index + round)`.
- Independent OptiMind validation for every returned candidate.
- Cost = min(solve-wall, 1.0 s) for terminal outcomes; otherwise 1.0 s.
- No feature or OptiMind planning overhead is charged to native `choose`, because the purpose is to test whether the solver's own default route avoids external feature cost.

## Pre-registered interpretation gates

M28 does **not** promote a learned selector.

The native-choose route is considered to close the portfolio opportunity only if all hold:

1. official corpus integrity true;
2. 48 unique fresh instances and zero overlap with M25;
3. complete 3-round accounting;
4. all three routes terminal rate >= 90%;
5. zero independent-solution invalidity;
6. zero material objective mismatch among routes when optimal objectives are available;
7. `choose / SBS <= 1.02` by mean cost;
8. paired-bootstrap 95% upper bound for `choose / SBS` < 1.05;
9. `choose` closes >= 80% of the SBS-to-VBS gap, when that gap is positive and >= 1%;
10. no evidence that `choose` is secretly a distinct cross-instance selector may be claimed unless algorithm-use diagnostics show both simplex and IPM are selected on non-presolve-trivial cases.

If `choose` is effectively simplex, M28 will say so. That is not a failure of HiGHS; it means the internal default route is a baseline rather than an instance-wise portfolio selector in this environment.

## Claims boundary

- M28 evaluates a fresh public/OOD LP-relaxation cohort only.
- Bundled HiGHS 1.10.0 is verification-only provenance and must not be represented as current public highspy 1.15.x behavior.
- Current HiGHS documentation is used only to contextualize present-day solver-option semantics.
- SciPy `method="highs"` is not called an automatic selector in M28 evidence for SciPy 1.17.0.
- No learned-selector test data from M27 are reused in the fresh M28 cohort.
