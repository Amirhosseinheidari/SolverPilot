# SALVAGE-S5 — Explainability / Decision Reporting

S5 salvages the claim-safety semantics of the legacy OptiMind decision-reporting layer without importing its report model, adapters, free-text claim classifier, approval workflow, or product-readiness bureaucracy.

## Design boundary

- `SolveResult` remains authoritative.
- `solverpilot.reporting.explain_result()` is a read-only adapter over structured solve evidence.
- Top-level `solverpilot.__all__` is unchanged.
- `VALID_OPTIMAL` is described as backend-reported optimality plus validated primal feasibility unless `OptimalityEvidence.independently_verified_optimal` is true.
- Feasible is never escalated to optimal.
- Planner choice is never escalated to solver superiority.
- Reuse is never escalated to a speedup claim.
- Backend infeasibility/unboundedness reports are explicitly distinguished from independent proofs.
- No arbitrary free-text claims are accepted or classified in S5.

## Renderers

JSON output is JSON-safe and rejects NaN/Infinity through the model boundary. Markdown escapes raw HTML, table pipes, and embedded newlines.
