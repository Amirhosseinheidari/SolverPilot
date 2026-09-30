# Ordered hardening evidence

These measurements are scoped development evidence, not blanket performance or
industrial reliability claims. Raw third-party MPS files remain outside Git.

| Check | Observation | Scope |
|---|---|---|
| Exact arithmetic | Dyadic/reference verifier median ratios 0.575, 0.523, 0.563 | Three planted bounded LP sizes, three paired repetitions; exact check outputs agree; not full-solve speed |
| Extended recovery | Coupled equality tests pass; unitcal_7 verified; pilot4 incomplete | Consumed public diagnostic replay, not held-out performance |
| Windows repeated calls | 35/35 LP and 35/35 QP independently verified | Small fixed models, post-warmup RSS range below 0.2 MB |
| Linux/WSL repeated calls | 35/35 LP and 35/35 QP independently verified | Same physical laptop, not a second hardware platform |
| Fresh synthetic gain guard | 24/24 held-out calls verified in each route; no candidate admitted | 24 training/12 test models, two repeats; guard 0.02070 s vs production 0.01987 s mean API cost |

The synthetic result does not show a speedup: guard/production is approximately
1.042. All cases fall back to production, with additional guard cost. No parameter
was retuned using those held-out results. The earlier preliminary synthetic run
is not substituted for this final result.

The first Windows QP soak had two invalid cold OSQP candidates. Tighter native
solve tolerances fixed this; validator tolerances were not relaxed. Enabling
polishing was also tested but emitted unsolicited text into the JSON health CLI,
so the final built-in setting tightens tolerances without enabling polishing.

The first Linux soak lacked optional OSQP and failed its gate. The missing package
was installed before the reported run. The script now checks that prerequisite
before starting and retains each result's error/status. Observations do not claim
absence of memory leaks or automatic native factorization reuse.

Very short deadline tests report cleanup latency beyond their requested budgets;
late results are discarded. Five-second calls completed with checked candidates.
The isolated API does not promise zero deadline overshoot or real-time scheduling.

## Provenance

- `certificate-speed.json`: step-1 source snapshot, commit `775cb5d`; verifier-only
  comparison against independently written Fraction operations.
- `extended-recovery.json`: step-2 original-model replay; nonfinite diagnostic
  quantities are strings and are never interpreted as successful bounds.
- `operations-windows.json` and `operations-linux.json`: final built-in OSQP
  configuration, one worker, 35 repetitions per LP/QP path.
- `synthetic-gain/`: fixed protocol, disjoint hashes, training measurements, model,
  gain guard, raw observations and final summary. Source and environment bindings
  are retained in the artifacts; a changed implementation invalidates the guard.

Historical evidence from previous releases and public challenges is unchanged.
All observations retain their original environment records and timing scope.

## GitHub platform evidence

Run `36705960396` on commit `c72693e` passed all 12 jobs. The nine OS/Python
cells (Linux, Windows, macOS; Python 3.12/3.13/3.14) each independently verified
all 70 soak calls. Their largest post-warmup RSS range was 278,528 bytes. These
are short controlled observations, not long-duration leak qualification.

`ci/` contains each platform's raw operational report and JUnit count summaries.
The optional-integrations job passed 1,495 tests with 20 skips. The local Windows
coverage run passed 1,462 with 49 skips before the four additional benchmark
accounting tests, which also passed. Combined local statement/branch coverage
was 81.47%; skipped prerequisites are not counted as successful executions.

## Public diagnostic challenge (promotion rejected)

`public-gain/` preserves 192 training and 144 test calls from runner commit
`c72693e`, including fixed protocols, model/guard digests and both cohorts. The
old declared grouping passed initially, but a subsequent review against the
[official Netlib notes](https://www.netlib.org/lp/data/readme) found related
GREENBEA/GREENBEB instances across training and test. No outcome was deleted or
used to retune the model. `audited-summary.json` is authoritative for promotion;
the original `summary.json` remains as an unmodified numerical report.

Of 48 test calls per route, independently verified within budget were 24 guarded,
22 default and 24 production. There were no verified objective mismatches and
zero actual guarded switches. API PAR10 guarded/production was 1.0013, with
bootstrap interval [1.0004, 1.0034]; guarded/default was 0.9289, interval
[0.7601, 1.0078]. These descriptive figures are not unbiased generalization
estimates because of the family overlap, and the numerical promotion gate
already failed. Automatic routing remains off.

Future corpus preparation and qualification now apply curated Netlib aliases
before collecting outcomes. The failed cohort is consumed and cannot become
a fresh evaluation set by deleting its related case. A new independently
audited cohort is required before any future promotion.
