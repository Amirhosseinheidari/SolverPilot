# PROVENANCE M13

**Version:** 0.0.14  
**Date:** 2026-08-31

M13 continues from the complete M12 source artifact and adds direct solver-core verification paths. No public solver wheel was substituted or mislabeled.

## Direct C-API solver provenance

The C-API verification backends are accepted only for the exact CasADi 3.7.2 installation and exact bundled solver identities verified at runtime.

### OSQP

- solver core version: `0.6.3`
- bundled by: `casadi 3.7.2`
- library SHA-256: `6e69e9bd09cd9db73d410e61af1f1eb409c161d892a61911b2b4fd1480a257a3`
- primary header SHA-256: `fd7243e3907e26618f8965954a9cba3c29d16a3622ca96696bb8503972b3cc6b`
- ABI configuration is checked before availability is granted, including integer/float widths and relevant structure offsets.

### HiGHS

- solver core version: `1.10.0`
- bundled by: `casadi 3.7.2`
- library SHA-256: `bce96689405f83a849500fa3cf680d61b3eeb2ab179d358726e86526a047df4f`
- C API header SHA-256: `b5384c3f1169c8cb09d21f9bc4e14c55bb3b62654c03ec87cd8ae47848365593`
- `HighsInt` verified as 4 bytes for this bundled build.

Both backends set `verification_only=True`, `public_python_package=False`, and are excluded from OptiMind's default production registry.

## Public package boundary

The project still targets dedicated public integrations separately:

- `highspy 1.15.1`
- `osqp 1.1.3`
- `PySCIPOpt 6.2.1`

The locked wheel URLs and expected SHA-256 values remain in `benchmarks/native-wheel-lock.json`. M13 attempted those URLs again but DNS/egress prevented download, so no public-package execution claim is made.

## Final release wheel

`dist/optimind_core_codename-0.0.14-py3-none-any.whl` was rebuilt **after** the final HiGHS global-scheduler fix and independently installed outside the source tree before release.
