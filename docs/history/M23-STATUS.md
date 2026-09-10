# M23 Status — Production Evidence-Aware Planner

**Version:** 0.0.28  
**State:** VERIFIED / COMPLETE

## Goal

M23 turns OptiMind's planner from a capability-only chooser into a production-oriented, proof-safe evidence consumer. It deliberately does **not** infer solver-speed superiority from M22, because the official M22 gate validated corpus identity, execution accounting, and selected reference checks rather than a multi-backend held-out selector study.

## New behavior

- explicit evidence classes: none / capability / corpus-validated / comparative-heldout;
- M22 gate parser converts real M22 evidence into `corpus_validated` evidence;
- performance ranking remains disabled for M22 evidence;
- LP conservative baseline: `scipy-highs-ds` when eligible;
- MILP conservative baseline: `scipy-highs-bridge` when eligible;
- QP conservative baseline: available continuous QP route, but non-certifying routes are rejected under `PROVE_OPTIMAL`;
- performance overrides are accepted only with comparative held-out evidence that includes at least two backends, fixed-environment accounting, and feature-cost accounting;
- performance overrides must still pass capability, availability, health, and solve-intent gates;
- `solve_production(...)` returns both the solve result and the auditable production decision.

## M22 evidence interpretation

The official M22 corpus gate passed for MIPLIB/QPLIB/PACE, with hash integrity, three repetitions, explicit timeout accounting, no read/worker errors, no reported reference mismatches, no invalid PACE candidates, plus MIPLIB and QPLIB reference cross-checks. M23 treats this as strong correctness/robustness evidence but **not** comparative solver-ranking evidence.

## Verification

- full regression: **229 collected / 226 passed / 3 skipped / 0 failed / 0 errors**;
- planner campaign: **900 / 900 decisions**, 0 failures;
- production solve campaign: **90 / 90 independently valid**;
- MILP brute-force objective matches: **30 / 30**;
- QP analytic-target matches: **30 / 30**;
- automatic performance ranking enabled from M22 evidence: **0 / 900**;
- invalid speed override requested from M22 evidence: rejected as designed;
- wheel 0.0.28 isolated install: PASS;
- wheel production LP/MILP/QP smoke: PASS;
- QP `PROVE_OPTIMAL` with only non-certifying routes: rejected as designed.

## Claim boundary

M23 is a production safety/orchestration milestone. It does not claim a learned selector, VBS-gap closure, or runtime superiority. A future performance selector needs a comparative multi-backend held-out benchmark with feature/inspection cost included.
