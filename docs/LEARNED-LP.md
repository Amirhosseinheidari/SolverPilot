# Experimental learned LP selection

This development API trains and evaluates a small cost-sensitive selector for
continuous LP. It does **not** change `solve()`, `solve_production()`, the default
registry, or GPU routing. It is not in the published 0.4 wheel.

The historical M26/M27/M29 experiments did not establish production-quality LP
routing. Their test outcomes have already been consumed and must not be reused as
fresh held-out evidence. This implementation starts a separate local experiment
with new generated models; synthetic results cannot qualify public/OOD workloads.

## What is implemented

- Eight inexpensive sparse structural features, without running a solver or
  materializing a dense matrix: variable/row/nonzero counts (log1p), density,
  equality/bounded-variable/objective-nonzero/positive-coefficient fractions.
- A cost-sensitive decision stump. Training searches feature thresholds and two
  candidate choices, minimizing total measured penalized time. Four distinct
  training groups per leaf and a 5 ms switching margin are defaults. A stump has
  one split; this deliberately bounds model complexity and inference cost.
- A single best candidate chosen **only on training data** as the fallback and
  evaluation baseline. Test outcomes never pick this baseline.
- All repetitions count. An independently verified solve completed within the
  cutoff costs its actual total wall time. Each failed, unverified, or late repeat
  costs ten times the cutoff (PAR10). Fast successes do not erase failed repeats.
- JSON model persistence with a content checksum, feature schema, training
  identities, group IDs, data hashes, environment and protocol binding. No pickle
  or scikit-learn runtime dependency. Checksums detect accidental modification;
  they are not authentication of a model's author or measurement truthfulness.
- Abstention outside the training feature ranges, on an environment mismatch, or
  when a selected candidate is unavailable. A missing fallback returns `None`.
  Feature ranges are a guard, not a calibrated OOD detector.
- A held-out research gate with measured feature/decision overhead, group-cluster
  bootstrap, at least six groups, at least two switches, >=3% mean PAR10 gain,
  bootstrap upper ratio <1, p90 slowdown <=25%, and no lost verified solves per
  instance relative to the frozen baseline.

## Research API

Import from `solverpilot.experimental.learned_lp`. Stable top-level APIs are
unchanged. Candidate IDs must encode solver algorithm/settings, since two HiGHS
configurations share the same adapter name.

```python
from solverpilot.experimental.learned_lp import (
    LPObservation, LPSelector, lp_features, fit_lp_selector,
    decide_lp_backend, evaluate_lp_selector,
)

# training_rows: LPObservation records with split="train", complete paired
# repetitions, canonical model hashes, and original-space independent validation.
model = fit_lp_selector(
    training_rows,
    candidates=("highs-simplex", "highs-ipm", "ortools-pdlp"),
    cutoff_s=2.0,
    protocol_sha256=frozen_protocol_sha256,
)
model.save("lp-selector.json")
model = LPSelector.load("lp-selector.json")
decision = decide_lp_backend(
    problem, model,
    environment_id=current_environment_id,
    available=available_candidate_ids,
)
# decision is a shadow recommendation, never a certificate or routing authority.
```

`decide_lp_backend` measures extraction and selection time. It does not measure
model loading, environment capture, candidate discovery or process startup; these
are amortized setup in the supplied campaign. A per-call deployment must account
for any additional costs it actually incurs. Candidate availability and environment
IDs are caller-supplied contracts, not automatic hardware/package detection.

`evaluate_lp_selector` rejects training/held-out overlap by name, group or canonical
data hash, mixed environments/splits, incomplete candidates/repeats, and decisions
inconsistent with the frozen model. It accepts observations of one held-out split
at a time. The caller must keep validation and test groups disjoint; the supplied
runner freezes their complete assignment before solving. Related instances that
are not byte-identical must have the same caller-assigned group.

The research gate always reports `production_authorized=False`, including a
positive synthetic result. Existing correctness/intent/health checks remain
mandatory. A shadow prediction does not establish feasibility or optimality.

## Reproduce the local experiment

Use a fresh output directory; the runner refuses to overwrite a campaign. Install
the development source with `[highs,pdlp]`, and set common BLAS thread limits to one.

```bash
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 \
NUMEXPR_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 BLIS_NUM_THREADS=1 \
python benchmarks/qualify_learned_lp.py --output fresh-lp-routing-run
```

The runner uses three families (planted sparse ranged LP, separable bounded LP,
transportation LP), ten independently seeded groups per family, and two variants
per group. Six groups per family train, two validate and two remain sealed for
test. Every candidate gets two serial solves with balanced order, one solver
thread and a 2-second solver budget. Total `solve()` wall time includes model
conversion and independent validation; times beyond 2 seconds are still recorded
and penalized. This is not a hard controller deadline for in-process HiGHS.
Warm-up is separate and recorded; PDLP's isolated worker is cold on every call.

The protocol, source hashes, environment, model identities and split manifest are
written before the first solve. The fitted model and each held-out prediction are
saved before the corresponding outcomes. Validation failure leaves the test
unsolved. Do not tune the model against this validation result and present the
same cohort as fresh evidence. Group bootstrap handles repeated family instances
through the declared groups; six validation groups still provide limited power.

See [the recorded local experiment](evidence/learned-lp-local/README.md) for results
and remaining qualification work. GPU is excluded from this campaign after the
previous CPU-PDLP comparison showed no GPU benefit at these sizes.

## Production gate repairs

`ProductionEvidence.supports_performance_ranking` now also requires corpus
integrity, complete outcome accounting, independent validation and reference
cross-check flags. A requested backend must belong to the evidence's compared
candidate set and pass the existing capability/intent/health checks. Rejected or
unused overrides report `auto_performance_ranking_enabled=False`.

These flags are a caller-declared evidence contract, not cryptographic verification
of an external report. No synthetic experiment or model file automatically sets
them. Real representative workloads, untouched groups, environment replication,
end-to-end overhead and tail/failure checks are still required before deployment.

The split discipline follows the [scikit-learn guidance on data leakage](https://scikit-learn.org/stable/common_pitfalls.html)
and [group-aware evaluation](https://scikit-learn.org/stable/modules/cross_validation.html).
