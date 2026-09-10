# SALVAGE-S4 — Feature Schema + OOD / Distribution Shift

S4 adds versioned feature contracts and conservative, explainable distribution-shift diagnostics without creating a second inspector or holdout system.

Design boundaries:

- `solverpilot.inspect.inspect_problem` remains the single source of the canonical `ProblemFingerprint`.
- `solverpilot.benchmark.splits` remains the authoritative train/validation/test and group-leakage contract.
- `solverpilot.intelligence` standardizes feature schema/records, audits pre-solve leakage, and fits shift profiles from training records only.
- OOD evidence is descriptive and fail-closed; it is **not** model confidence and does not authorize production solver ranking.
- No selector or production-routing behavior changes in S4.
- The frozen top-level `solverpilot` API remains unchanged; S4 is exposed under `solverpilot.intelligence`.
