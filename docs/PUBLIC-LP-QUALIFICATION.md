# Public workload qualification of the frozen LP selector

This experiment challenges the unchanged synthetic-trained selector on unused
public LP instances, comparing **actual routed calls**, not hindsight replay.
Automatic learned routing remains disabled. No package release is implied.

## Admission before solving

`benchmarks/prepare_public_lp.py` uses the official
[Netlib LP table](https://www.netlib.org/lp/data/readme) and
[MIPLIB benchmark metadata](https://miplib.zib.de/tag_benchmark.html).
It excludes instances in the audited M24/M25/M28 cohorts and additional local
qualification reports. Published MIPLIB groups exclude related known families;
Netlib uses name-prefix grouping. The record documents the history audited, not
a guarantee about unrecorded or renamed outside datasets.

Selection is by SHA256(source:name), with 12 admitted groups per source and caps
of 50,000 variables, 50,000 constraints and 500,000 nonzeros. It does not use solver
outcomes or published difficulty/status. Download/import failures remain in the
record. No instance is replaced after a difficult solve.

Netlib's official `emps.c` expands its compressed MPS files. Both SolverPilot and
native HiGHS independently read each MPS. Matrix entries, objective, offset, sense
and all bounds must agree exactly. MIPLIB integer domains are then relaxed to
continuous domains, preserving bounds. Published **integer** objectives are not
LP references. Netlib's published LP optima provide an additional cross-check;
numerical proof status still requires original-space independent validation.

Raw third-party instances are not redistributed in Git. Source URLs, compressed
and expanded byte hashes, canonical model hashes, dimensions, reader agreement
and all admission exclusions are preserved.

## Frozen execution

The selector is the original JSON model from the local experiment. No retraining,
threshold changes, new features or result-based tuning is allowed. Six strategies
are compared: the frozen selector, ordinary `solve()`, ordinary
`solve_production()`, native HiGHS simplex, native HiGHS IPM and CPU OR-Tools PDLP.

Each strategy gets two serial repetitions, in rotated/reversed order, using fresh
processes. One logical CPU affinity is inherited by descendants; common BLAS
environment limits are one. Explicit candidates request one solver thread.
Default APIs retain normal thread settings: affinity limits actual CPU execution,
not thread creation. This is an isolated-call profile, not a persistent server.

The 2-second API budget includes environment discovery, model loading and selector
decision overhead where applicable; remaining time is passed to the solver.
Model import/canonicalization are common pre-call work. Both API time and full
parent-observed process wall time are recorded. A separate 12-second controller
deadline kills the entire worker process group, including nested PDLP workers.
The native solver budget does not itself bound independent-validation latency.

For primary PAR10 scoring, a solve must have independently verified optimality,
API time <=2 seconds and controller time <=12 seconds. Its cost is **full process
wall time**; otherwise that repetition costs 120 seconds. Every unsuccessful,
unverified or late repeat counts. This is penalized completion cost, not average
successful-solve latency. Raw timings distinguish verified-but-late outcomes.

Acceptance requires complete outcomes, >=20 groups, actual in-support learned
switches, no material disagreement among verified objectives or with Netlib
references, and all these conditions against **both** default APIs:

- >=3% lower mean penalized cost.
- 95% group-bootstrap ratio upper bound <1 (2,000 draws, seed 271828).
- p90 per-instance slowdown <=25%.
- No lost verified-within-budget repetitions on any instance.

All other comparisons are retained even when unfavorable. Passing a comparison
would still require deployment scope/profile review; it never automatically
enables production routing.

## Environment binding repair

WSL reported 8 KiB more guest RAM than during training on the same laptop, changing
the old exact environment fingerprint. `experimental.lp_environment` binds the
current measurements to the hash-verified original training protocol, tolerating
only <=1 MiB absolute positive-RAM reporting drift. Other fingerprint fields and
observed thread settings must match. Actual and training IDs, delta and reason
are recorded; neither observations nor the model are rewritten. Rejected thread
settings get a distinct rejected decision ID even if the legacy raw fingerprint
omitted that difference.

This repair was frozen before public solves. The stump, training baseline and
feature-support ranges are unchanged. Affinity and cold-call timing are a distinct
public execution profile; compatibility does not make synthetic/public timings
interchangeable.

## Stress tests and limits

Separate deterministic cases exercise infeasibility, unboundedness, ill-scaling
and a tiny deadline. They are not used for training or public performance scoring.
Infeasible/unbounded backend terminations are never counted as independently
verified optimality. Optional termination-certificate recovery is not activated.

The frozen policy is executed without posthoc solver retries. Its pre-existing
abstention uses the training SBS (PDLP), which may be a poor public-workload
fallback. Any changed fallback or retrained model needs a new untouched cohort.

## Reproduction

Use the development checkout with `[highs,pdlp]` on Linux/WSL, captured official
metadata, the prior-history audit and a separately built Netlib decoder. Output
directories must be new; the runner refuses to overwrite consumed evidence.

```bash
PYTHONPATH=src python benchmarks/prepare_public_lp.py \
  --metadata captured-metadata --output public-corpus --decoder /path/to/emps

PYTHONPATH=src OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
python benchmarks/qualify_public_lp.py \
  --corpus public-corpus --model docs/evidence/learned-lp-local/model.json \
  --output public-qualification
```

See [the recorded experiment](evidence/public-lp-local/README.md) for the exact
protocol, cohort, outcomes, stress results and decision.
