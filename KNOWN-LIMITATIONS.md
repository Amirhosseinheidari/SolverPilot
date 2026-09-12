# Known limitations — 0.4

Version 0.4 adds explicit global optimization, generalized power cones, CPU PDLP and conservative certificate recovery. See the [0.4 guide](docs/release/SOLVERPILOT-PUBLIC-DOCS-0.4.md) for exact scope and APIs.

## Numerical evidence

Primal validation is independent of backend termination. Continuous LP/QP duals from SciPy LP, native HiGHS and OSQP can additionally be checked for stationarity, complementarity and numerical gap. These are tolerance-qualified checks, not exact-arithmetic proofs. MILP/global nonlinear optimality is not proved by primal feasibility. Backend-only infeasible/unbounded statuses remain claims unless a certificate is checked.

Per-call ValidationTolerances is supported by execute, solve, solve_production and execute_portfolio. The default feasibility floor is 1e-7 with relative tolerance 1e-9; the relative allowance can be disabled. Native solver tolerances must be configured separately. Nonfinite evidence cannot produce a verified certificate.

## Conic and nonlinear scope

Optional direct Clarabel supports convex quadratic objectives and affine SOC, rotated SOC, PSD, exponential and power cones. Runtime conformance and original-space validation gate its use. CasADi SuperSCS retains its restricted linear-objective SOC/RSOC scope; its generic PSD and quadratic-conic routes remain disabled. Generalized power cones are supported by direct Clarabel. Conservative independent SOC/RSOC/PSD bounds require exact dual membership, objective PSD evidence and a finite residual correction; transcendental cone dual certificates remain unsupported. Conic primal feasibility alone does not imply independently verified global optimality.

Smooth NLP remains optional through CasADi/Ipopt and produces local candidates. An explicit SCIP path now accepts bounded factorable global NLP/MINLP, general integers, nonlinear equalities/two-sided rows and supported abs/max/min atoms. Indicator bodies require provable finite interval bounds. Arbitrary Python functions and independent generic global proofs remain unsupported. The existing convex binary OA path retains its previous restrictions and proof semantics.

## Execution and ownership

Batch accepts canonical LP/QP, uses spawn and requires a script main guard. Timeouts/cancellation terminate workers; unobserved incumbents cannot be recovered afterward. Native solver budgets may return incumbents before the outer deadline. Cleanup may add bounded latency. POSIX memory limits are RLIMIT_AS address-space caps, not RSS; Windows memory caps are rejected. Worker count does not limit internal solver threads.

ReoptimizationSession owns a model clone and serializes update+solve. Session/PersistentSession serialize their public methods, but direct external model/backend mutations require caller coordination. Compiler caches are not shared mutable workspaces for uncoordinated modeling. Reuse is conditional on structure/settings and is not guaranteed faster. Session history can intentionally grow.

PDLP supports only continuous LP/nonnegative diagonal QP and is process-isolated. It is not automatically selected. Exact SCIP and GPU readiness probes do not qualify execution; neither corresponding execution adapter is shipped.

CP-SAT remains process-isolated to avoid known HiGHS ABI collisions. Reference CP and exact TSP/VRP are exhaustive and only suitable for small oracle cases.

## Modeling and records

Soft helpers accept affine relations and require explicit penalty use in the objective. Lexicographic solving currently supports canonical LP/MILP with linear objectives; locks include numerical tolerances. Named conflict diagnostics support LP/MILP. Named values reject stale compiled models and retain IDs for duplicate labels.

Learned routing remains disabled; OOD indicators are not calibrated probabilities. History is opt-in plaintext SQLite; user metadata can be sensitive. Extension registry mutation requires caller coordination. Development pickle caches are trusted artifacts, not a safe untrusted interchange format.

## Qualification

Cross-platform support requires exact-artifact qualification on Python 3.12–3.14. Upstream package availability alone is not integration evidence. Synthetic latency/RSS measurements do not establish industrial scalability or absence of long-term leaks. Historical reports retain their original versions and cannot qualify new artifacts.

See [the 0.4 guide](docs/release/SOLVERPILOT-PUBLIC-DOCS-0.4.md) for APIs and examples.
