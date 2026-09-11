# Known limitations — 0.2

Version 0.3 adds checked convex atoms, sparse affine lowering, bounded streaming, sensitivity and scenarios. See the current [0.3 guide](docs/release/SOLVERPILOT-PUBLIC-DOCS-0.3.md) for guarantees and remaining scope limits.

## Numerical evidence

Primal validation is independent of backend termination. Continuous LP/QP duals from SciPy LP, native HiGHS and OSQP can additionally be checked for stationarity, complementarity and numerical gap. These are tolerance-qualified checks, not exact-arithmetic proofs. MILP/global nonlinear optimality is not proved by primal feasibility. Backend-only infeasible/unbounded statuses remain claims unless a certificate is checked.

Per-call ValidationTolerances is supported by execute, solve, solve_production and execute_portfolio. The default feasibility floor is 1e-7 with relative tolerance 1e-9; the relative allowance can be disabled. Native solver tolerances must be configured separately. Nonfinite evidence cannot produce a verified certificate.

## Conic and nonlinear scope

Optional direct Clarabel supports convex quadratic objectives and affine SOC, rotated SOC, PSD, exponential and power cones. Runtime conformance and original-space validation gate its use. CasADi SuperSCS retains its restricted linear-objective SOC/RSOC scope; its generic PSD and quadratic-conic routes remain disabled. Generalized power cones are not implemented. Conic primal feasibility alone does not imply independently verified global optimality.

Smooth NLP remains optional through CasADi/Ipopt and produces local candidates. Generic global NLP and arbitrary nonsmooth atoms are unsupported. MINLP remains the certified convex binary-discrete scope: general integer, nonconvex/global, nonlinear-equality, two-sided nonlinear and indicator+MINLP compositions remain unsupported. Existing proof flags retain their documented solver-certified scope.

## Execution and ownership

Batch accepts canonical LP/QP, uses spawn and requires a script main guard. Timeouts/cancellation terminate workers; unobserved incumbents cannot be recovered afterward. Native solver budgets may return incumbents before the outer deadline. Cleanup may add bounded latency. POSIX memory limits are RLIMIT_AS address-space caps, not RSS; Windows memory caps are rejected. Worker count does not limit internal solver threads.

ReoptimizationSession owns a model clone and serializes update+solve. Session/PersistentSession serialize their public methods, but direct external model/backend mutations require caller coordination. Compiler caches are not shared mutable workspaces for uncoordinated modeling. Reuse is conditional on structure/settings and is not guaranteed faster. Session history can intentionally grow.

CP-SAT remains process-isolated to avoid known HiGHS ABI collisions. Reference CP and exact TSP/VRP are exhaustive and only suitable for small oracle cases.

## Modeling and records

Soft helpers accept affine relations and require explicit penalty use in the objective. Lexicographic solving currently supports canonical LP/MILP with linear objectives; locks include numerical tolerances. Named conflict diagnostics support LP/MILP. Named values reject stale compiled models and retain IDs for duplicate labels.

Learned routing remains disabled; OOD indicators are not calibrated probabilities. History is opt-in plaintext SQLite; user metadata can be sensitive. Extension registry mutation requires caller coordination. Development pickle caches are trusted artifacts, not a safe untrusted interchange format.

## Qualification

Cross-platform support requires exact-artifact qualification on Python 3.12–3.14. Upstream package availability alone is not integration evidence. Synthetic latency/RSS measurements do not establish industrial scalability or absence of long-term leaks. Historical reports retain their original versions and cannot qualify new artifacts.

See [the 0.2 guide](docs/release/SOLVERPILOT-PUBLIC-DOCS-0.2.md) for APIs and examples.
