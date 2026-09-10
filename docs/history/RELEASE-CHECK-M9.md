# RELEASE CHECK — M9

**Version:** 0.0.10

- source compileall (`src`, `tests`, `benchmarks`, `tools`): PASS
- final source pytest: 157 passed / 3 skipped / 0 failed/errors
- wheel build: PASS
- wheel: `dist/optimind_core_codename-0.0.10-py3-none-any.whl`
- wheel size: `72959` bytes
- wheel SHA-256: `27365d5bdbe4d35f65a2fdefa329b433981d89813544044c258ee6a6d0bf9f2d`
- isolated target install (`--no-deps --no-index`): PASS
- installed wheel reports version 0.0.10: PASS
- installed wheel NLopt solve + independent validation: PASS
- installed wheel NLopt active health probe: PASS
- installed wheel `PROVE_OPTIMAL` abstention when only no-certificate QP backends exist: PASS
- installed wheel QP portfolio: exactly 2 requested backends -> exactly 2 attempts, no hidden errors: PASS
- installed wheel backend-health JSON generation: PASS

The target install intentionally reused the runtime's already installed NumPy/SciPy/NLopt dependencies while isolating the OptiMind package itself from the source tree.
