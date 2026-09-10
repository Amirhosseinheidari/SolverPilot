# INTERNET / API AUDIT M13

Date: 2026-08-31

M13 re-checked solver lifecycle/update semantics against official sources and the exact bundled headers used for ABI verification.

## OSQP 0.6.3 C API

Official release-0.6.3 documentation confirms:

- `osqp_setup` creates the workspace and performs factorization/preconditioning;
- OSQP automatically warm-starts primal/dual variables from the previous QP solution;
- `osqp_update_lin_cost` and `osqp_update_bounds` update vectors without re-setup;
- `osqp_update_P`, `osqp_update_A`, and `osqp_update_P_A` update matrix values **without changing sparsity structure**.

Source: https://osqp.org/docs/release-0.6.3/interfaces/C.html

The current OSQP migration guide also confirms that these 0.6.x names/signatures changed in 1.x, which is why M13's C ABI bridge is pinned to solver core 0.6.3 and must not be reused for OSQP 1.x.

Source: https://osqp.org/docs/get_started/migration_guide.html

## HiGHS C API lifecycle

HiGHS documentation confirms:

- HiGHS has a **global scheduler shared across HiGHS instances**;
- `Highs_destroy` does not free that scheduler;
- `Highs_resetGlobalScheduler` is required to release scheduler resources and before changing thread count across independent runs;
- reset while another HiGHS instance is actively in use is undefined behavior;
- model clocks are cumulative across repeated `Highs_run`, so `Highs_zeroAllClocks` should be called before an independently timed repeated solve.

Source: https://ergo-code.github.io/HiGHS/dev/interfaces/c_api/

The exact bundled HiGHS 1.10.0 header shipped with CasADi 3.7.2 contains both `Highs_resetGlobalScheduler` and the `Highs_zeroAllClocks` cumulative-time warning. M13 hashes that header and shared library in backend metadata.

## CasADi provenance

The direct C-API backends are accepted only when installed CasADi version is exactly 3.7.2 and the expected bundled solver versions/header/configuration/ABI checks match. They remain verification-only.

Official CasADi downloads page: https://web.casadi.org/get/

## Public Python packages

M13 retains the wheel lock for public `highspy 1.15.1`, `osqp 1.1.3`, and `PySCIPOpt 6.2.1`. Direct file-host attempts still fail because this runtime cannot resolve `files.pythonhosted.org`; this is recorded in `benchmarks/results/m13-native-wheel-network-attempt.json`.
