# SolverPilot Public API Reference

The supported top-level API is exactly the set in `solverpilot.__all__`. The surface was originally frozen at M30; the current prerelease snapshot is [`PUBLIC-API-V0_1_0RC2.json`](../../PUBLIC-API-V0_1_0RC2.json).

## Stability policy

- Top-level names in `solverpilot.__all__` are the compatibility surface targeted for the 1.x line.
- Removing or changing the meaning of a frozen top-level symbol requires a major-version change once 1.0 is released.
- New optional parameters may be added when backward compatible.
- `solverpilot.experimental` has **no** compatibility guarantee.
- Internal modules, benchmark scripts, milestone evidence formats, and private helpers are not public API unless explicitly listed.

## Problem/model API

`LinearProblem`, `QuadraticProblem`, `ObjectiveSense`, `VariableDomain`, `ConvexityStatus`, `parse_mps`, and `read_mps` define the canonical supported model surface.

## Solve API

- `solve()` — deterministic capability-aware solve; no hidden empirical performance claim.
- `solve_production()` — conservative evidence-aware routing plus an auditable `ProductionDecision`.
- `execute()` — execute one explicit backend and independently validate the result.
- `execute_portfolio()` — validated sequential fallback primitive; not a learned selector.

## Result/status API

`SolveResult` contains the public status and accepted candidate data plus auditable backend/runtime context:

- `status`: canonical `PublicStatus`;
- `objective`: accepted objective value, or `None` without a candidate/objective;
- `x`: accepted candidate vector, or `None`;
- `validation`: `ValidationReport` for candidate-bearing results;
- `diagnostics`: optional infeasibility diagnostics when requested;
- `backend_status`: solver-native termination text;
- `trace`: executed backend and phase timings;
- `plan`: automatic routing decision when planning was used.

`SolveResult.status` uses `PublicStatus`, while solver-native status strings remain available separately in `backend_status`. Candidate-bearing statuses are exposed only after canonical candidate validation. `INFEASIBLE`, `UNBOUNDED`, and `INFEASIBLE_OR_UNBOUNDED`, however, normally reflect backend termination conclusions unless independent certificate/diagnostic evidence is explicitly recorded; callers must not interpret those statuses as automatically independently certified.

## Stable error semantics

- `BackendUnavailableError`: an explicitly requested backend cannot be used in the active environment.
- `CapabilityMismatchError`: an explicitly selected backend cannot safely represent the supplied problem.
- `UnknownBackendError`: an explicit backend identifier is not registered.
- `BudgetNotSupportedError`: a requested wall-time/thread/memory budget cannot be enforced safely.
- `NoCompatibleBackendError`: planning cannot find any available backend satisfying the hard capability/intent/health gates.
- Solver termination such as infeasible, unbounded, limit, or solver error is normally represented in `SolveResult.status`, not raised as an exception.
- Invalid user data may continue to raise `ValueError`/`TypeError`; MPS parsing uses `MPSParseError` and `MPSUnsupportedFeatureError`.

## Custom backends

`Backend`, `BackendSolveResult`, `BackendManifest`, and `BackendRegistry` are frozen so advanced users can provide an explicit adapter without relying on private runtime internals. Every adapter result still passes through canonical validation.

## Experimental migration

Pre-M30 top-level imports such as:

```python
from solverpilot import production_evidence_from_m29_value_audit
```

are research-only. M30 provides a temporary `DeprecationWarning` compatibility shim; the supported location is:

```python
from solverpilot.experimental import production_evidence_from_m29_value_audit
```

## Extended prerelease submodule APIs

Track P P0–P9 adds modeling/conic/NLP/MINLP/CP functionality under explicit submodules without expanding the frozen 78-symbol top-level `solverpilot.__all__` surface in this merge candidate. See [Extended Modeling API](EXTENDED-MODELING-API.md).
