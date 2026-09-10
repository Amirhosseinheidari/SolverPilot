# Pre-Public Release Verification — SolverPilot 0.1.0rc2

This checkpoint supersedes the local `0.1.0rc1` candidate after adversarial correctness/trust hardening and remains separate from historical `0.0.40rc2` provenance. It does not authorize publication.

Local verification must pass the full source regression, current API/backend snapshot checks, build/package checks, isolated-wheel tests, and source hygiene. External GitHub qualification remains mandatory for the exact release commit and exact artifacts before PyPI publication.

Historical Track P merge provenance remains frozen at `0.0.40rc2`; current package identity is `0.1.0rc2`.

Local rc2 qualification additionally requires the permanent adversarial regression suite for QP canonicalization/convexity, conic scaling/PSD validation, non-finite rejection, immutable result evidence, and proof-aware benchmark identity.
Final local hardening also requires zero runtime `assert` statements under `src/solverpilot`, controlled CP timeout/invalid-worker-response failures, fail-closed MPS vector ambiguity, explicit separation of solver-certified versus independently verified MINLP proof, installed-wheel execution under `python -O`, release-workflow static/security/SBOM gates, and source-tree cleanup before the final source archive is frozen.

The local build environment may be used only for diagnostic wheel/sdist checks when it does not provide the exact build-system pins from `pyproject.toml`. Exact public-release qualification remains an external GitHub gate.
