# Known Limitations — 0.1.0rc2

This file records real limitations and guarantee boundaries for the merged SolverPilot + Track P prerelease.


## 0.1.0rc2 release-candidate boundaries

The S2–S10 migration layers are part of this prerelease source line, but the 78-symbol top-level API remains frozen. The new layers are intentionally submodule-scoped. Public release qualification still requires the exact-artifact GitHub matrix, including optional native integrations and artifact attestation before publication.

The learned solver selector remains disabled. OOD diagnostics are conservative shift indicators rather than calibrated probabilities. History persistence is opt-in plaintext SQLite and intentionally excludes raw solution vectors/problem arrays, but caller-provided metadata can still be sensitive. TSP/VRP exact reference solvers are exponential and intended only for small validation/oracle cases. Extension activation assumes caller-owned canonical registries are not mutated concurrently outside the manager during activation.

## Validation tolerances

Candidate validation uses `ValidationTolerances` defaults with an absolute feasibility floor of `1e-7` plus an internal scale-aware relative term. Raw absolute residuals are still reported, and a warning records when a candidate is accepted only by the scale-aware criterion. High-level `solve()` / `solve_production()` does not yet expose a per-call validation-tolerance parameter, so users with domain-specific numerical requirements should still validate against their own tolerances.

## Infeasible / unbounded status evidence

Candidate-bearing statuses are independently checked against canonical constraints, integrality, and objective consistency. `INFEASIBLE`, `UNBOUNDED`, and `INFEASIBLE_OR_UNBOUNDED` normally reflect backend termination unless independent certificate/diagnostic evidence is explicitly recorded.

## Learned LP routing

Learned LP performance routing remains disabled. Track P does not change or override the M24–M29 negative generalization decision.

## Conic boundary

SOC and RSOC representation/validation are supported, with a restricted verified linear-objective solve path. PSD representation/validation is supported, but generic PSD solving is not promoted as verified. Quadratic-objective conic solving also remains fail-closed without the necessary conformance evidence. Exponential/power/generalized-power cones are not implemented.

## NLP boundary

Smooth continuous NLP is optional and uses the pinned CasADi verification path. Generic NLP results are local-optimal candidates only after independent primal/objective validation and KKT stationarity checks. SolverPilot does not claim global optimality for generic NLPs. Nonsmooth generic atoms such as arbitrary `abs/max/min` are not part of the current verified surface.

## MINLP boundary

The current proof-aware MINLP layer supports a certified convex **binary-discrete** scope. General integer MINLP, nonconvex/global MINLP, nonlinear equalities, two-sided nonlinear constraints, and indicator+MINLP composition remain unsupported/fail-closed.

## Constraint-programming boundary

OR-Tools CP-SAT is executed in a Python isolated-mode subprocess rather than loaded after the core HiGHS path in the same process. This is intentional: the verified Linux OR-Tools 9.15.6755 wheel and the SciPy/HiGHS path can expose incompatible HiGHS shared-library ABIs when loaded in the wrong order. The worker protocol binds the request/response to the SolverPilot version, exact OR-Tools version, request nonce, structural hash, and data hash. Worker results cross a strict JSON boundary and primal/objective data are revalidated in the parent process; backend optimality is preserved only when the worker status/proof metadata and parent checks are mutually consistent.

The reference CP backend is exhaustive and can return `unknown_state_limit` when its configured state budget would be exceeded. OR-Tools CP-SAT is optional and locked to version `9.15.6755` for the frozen P9 integration evidence; public cross-platform support still depends on the release qualification matrix.

## Persistence / reoptimization

`PersistentSession` can select native patching only when exact granular capability evidence is runtime verified. Reuse is not guaranteed to be faster; Track P retained negative timing cases where a persistent solve was slower than a cold rebuild.

## Cross-platform release status

Local Linux validation is not equivalent to public support. Linux/macOS/Windows support for Python 3.12–3.14 is promoted only after the manual exact-artifact release qualification workflow passes.

## Optional integrations

Availability depends on third-party binary packages, bundled plugins, and platform support. An upstream wheel existing is not by itself SolverPilot integration evidence.

## Historical 0.0.40rc2 core hardening semantics

- A `QuadraticProblem` may still be constructed with `convexity_status=UNKNOWN` for compatibility, but it is not eligible for convex-QP capability classification, planning, inspection, runtime execution, or guarded direct QP backend solving until convexity is confirmed.
- `PublicStatus.VALID_OPTIMAL` remains a compatibility status: it means the backend reported optimality and the canonical primal candidate validated. Inspect `SolveResult.optimality_evidence` for whether dual/gap/certificate evidence was independently verified.
- Linear/QP primal feasibility now uses an absolute numerical floor plus a scale-aware relative term internally. Large absolute residuals accepted only by the scale-aware criterion are surfaced as validation warnings.
- NLP IR records domain hazards for `log`, `sqrt`, and symbolic division when safety cannot be proved from declared bounds. Compilation remains compatible; the Ipopt bridge uses deterministic finite starting-point search, and final original-space validation remains authoritative.
- MINLP `globally_proven=True` is solver-certified within the compiler's convexity assumptions. `independently_verified_global` is a separate field and is currently false for the P8 orchestration path because SolverPilot does not reconstruct the complete end-to-end KKT/MILP proof chain independently.


## Research benchmark cache security

Some scripts under `benchmarks/` consume locally generated Python pickle caches. These are trusted-development artifacts, not safe interchange files. Loading a pickle from an untrusted source can execute code. Public dataset acquisition and scientific evidence must use the hash/provenance-controlled acquisition path rather than accepting third-party pickle caches.


## Local build reproducibility boundary

With a fixed `SOURCE_DATE_EPOCH`, the current local diagnostic wheel was byte-reproducible across two independent builds, while the sdist had identical member names and semantic file contents but not identical archive bytes. This local environment also does not provide the exact build-system versions pinned by `pyproject.toml`. Public-release reproducibility and exact build-backend qualification therefore remain GitHub release-qualification gates rather than locally certified properties.
