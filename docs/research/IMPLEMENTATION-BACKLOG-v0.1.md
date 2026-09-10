# Implementation Backlog v0.1

**Status:** Ready for implementation planning  
**Date:** 2026-08-30

Priority definitions:
- P0 = blocks correctness/core architecture
- P1 = required for useful alpha
- P2 = useful after first benchmark
- EXP = experimental, blocked by evidence

---

## EPIC 0 — project identity / governance

- [P0] Keep “OptiMind” as codename only.
- [P0] Select a non-conflicting public name before PyPI/GitHub launch.
- [P0] Decide project license after dependency/legal review.
- [P1] Define supported Python versions and OS matrix.
- [P1] Define semantic versioning and trace-schema versioning.

Exit:
- public package identity does not collide materially with Microsoft OptiMind;
- license and dependency policy documented.

---

## EPIC 1 — canonical problem contracts

- [P0] `LinearProblem` immutable structural schema.
- [P0] ranged row/variable bounds.
- [P0] continuous/integer/binary domains.
- [P0] objective sense + offset.
- [P0] sparse matrix validation.
- [P0] `QuadraticProblem` with sparse P and convexity status.
- [P0] structural/data hashes.
- [P0] typed model mutations.

Tests:
- empty rows/columns;
- infinities;
- duplicate sparse entries normalization;
- NaN rejection;
- dimension mismatch;
- binary-bound normalization policy;
- objective reconstruction.

Exit:
- Tier-0 problems can be represented without backend imports.

---

## EPIC 2 — capabilities

- [P0] capability enum.
- [P0] `NATIVE/EMULATED_SAFE/EMULATED_RISKY/UNSUPPORTED/UNKNOWN`.
- [P0] backend manifest schema.
- [P0] required-capability extraction from problem.
- [P0] transformation policy.
- [P0] explicit failure on unsupported semantics.

Exit:
- planner cannot select an incompatible backend.

---

## EPIC 3 — normalized results and validation

- [P0] termination enum.
- [P0] solution schema.
- [P0] LP primal validation.
- [P0] MILP integrality validation.
- [P0] QP objective recomputation.
- [P0] validation tolerance policy.
- [P0] reported-vs-recomputed objective check.
- [P1] unbounded/infeasible evidence type schema.

Exit:
- no backend result bypasses validator.

---

## EPIC 4 — HiGHS adapter

- [P0] LP build/solve.
- [P0] MILP build/solve.
- [P0] convex QP build/solve.
- [P0] time/thread/seed option mapping where supported.
- [P0] status normalization.
- [P1] basis extraction/injection.
- [P1] MIP solution start.
- [P1] callback trace capture.
- [P1] model data mutation.
- [P2] IIS adapter marked experimental according to upstream status.

Exit:
- Tier-0 corpus passes;
- forced-runtime benchmark matches direct HiGHS semantics.

---

## EPIC 5 — OSQP adapter

- [P0] convex QP build/solve.
- [P0] status normalization.
- [P0] infeasibility certificate normalization.
- [P1] data update path.
- [P1] automatic previous-solution reuse/session integration.
- [P1] timing extraction: setup/update/solve/polish.
- [P1] sparsity-change invalidation.

Exit:
- repeated-QP benchmark has cold and reuse baselines.

---

## EPIC 6 — SCIP/PySCIPOpt adapter

- [P0] LP/MILP build/solve for v0.1.
- [P0] status normalization.
- [P1] solution injection.
- [P1] event/progress trace.
- [P1] safe mutation lifecycle using solver stages.
- [P1] IIS integration.
- [P2] controlled native reoptimization experiments.
- [P2] quadratic/nonlinear capability exploration for post-v0.1.

Exit:
- MILP Tier-0 passes;
- no mutation is applied in an invalid SCIP stage.

---

## EPIC 7 — inspector

- [P0] cheap matrix dimensions/density.
- [P0] variable-type fractions.
- [P0] bound/sense statistics.
- [P0] coefficient dynamic ranges.
- [P0] row/column nnz distribution.
- [P0] singleton counts.
- [P0] structural hash.
- [P1] QP symmetry/PSD verification policy.
- [P1] numeric warning rules.

Exit:
- deterministic fingerprint;
- feature timing recorded.

---

## EPIC 8 — planner v0

- [P0] installed backend discovery.
- [P0] hard capability filter.
- [P0] `SolveIntent`.
- [P0] `SolveBudget`.
- [P0] rationale generation.
- [P1] repeated-QP reuse preference.
- [P1] LP-basis reuse preference.
- [P1] MIP-start preference.
- [P1] fallback chain.
- [EXP] pilot race.

Do NOT add learned selection yet.

Exit:
- every planner decision is explainable and deterministic for fixed environment.

---

## EPIC 9 — Session

- [P0] versioned mutation log.
- [P0] structural-vs-data mutation distinction.
- [P0] reuse state registry.
- [P0] invalidation rules.
- [P1] HiGHS basis state.
- [P1] HiGHS MIP incumbent state.
- [P1] OSQP repeated state/update.
- [P1] SCIP solution/reoptimization experiment path.
- [P1] cold fallback.

Exit:
- randomized mutation tests never reuse stale/incompatible state.

---

## EPIC 10 — diagnostics

- [P1] static scaling warnings.
- [P1] typed infeasibility evidence.
- [P1] SCIP IIS.
- [P1] OSQP certificates.
- [P2] HiGHS experimental IIS.
- [P2] generic elastic relaxation.
- [P2] deletion-filter conflict minimization with hard budget.

Exit:
- diagnostic test corpus validates every returned explanation.

---

## EPIC 11 — trace/reproducibility

- [P0] trace schema.
- [P0] solver version.
- [P0] problem hash.
- [P0] parameters/seed/thread count.
- [P0] phase timings.
- [P0] validation record.
- [P1] hardware metadata.
- [P1] progress events.
- [P1] JSON/JSONL serialization.
- [P1] redactable user metadata.

Exit:
- a benchmark result can be audited without relying on console logs.

---

## EPIC 12 — benchmark harness

- [P0] Benchopt skeleton.
- [P0] Tier-0 dataset.
- [P0] direct solver baselines.
- [P0] forced-runtime baselines.
- [P1] QPLIB convex-continuous manifest.
- [P1] MIPLIB 2017 benchmark manifest.
- [P1] repeated-QP generator.
- [P1] repeated-LP generator.
- [P1] repeated-MILP generator.
- [P1] robustness corpus.
- [P1] SBS/VBS computation.
- [P1] gap-closure metric.
- [P1] clean-environment CI.

Exit:
- benchmark can be reproduced from a fresh environment and emits raw result files.

---

## EPIC 13 — optional Gurobi adapter

- [P1] detect availability/license without breaking core.
- [P1] LP/QP/MILP build.
- [P1] warm starts/MIP starts.
- [P1] callbacks.
- [P1] IIS.
- [P2] MIQP/QCQP/nonlinear expansion.

Exit:
- package imports and core tests succeed when Gurobi is absent.

---

## EPIC 14 — post-v0.1

- [P2] Clarabel backend.
- [EXP] cuOpt hardware-aware backend.
- [P2] Pyomo/CVXPY/MathOpt import adapters.
- [EXP] dynamic/pilot features.
- [EXP] learned selector.
- [EXP] parameter configuration policy.
- [EXP] CP/Scheduling semantic IR.
- [EXP] general conic IR.
- [EXP] NLP/MINLP.
- [EXP] black-box optimization engine.

None of these block v0.1.

---

## Suggested implementation order

```text
Problem IR
  -> capabilities
  -> validator/results
  -> HiGHS
  -> Tier-0 tests
  -> trace
  -> OSQP
  -> SCIP
  -> inspector
  -> Session
  -> planner
  -> benchmark harness
  -> diagnostics
  -> optional Gurobi
  -> run benchmark
  -> revise architecture from evidence
```

The key rule is: **do not tune the planner before the direct and forced-runtime baselines exist.**
