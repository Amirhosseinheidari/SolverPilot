# Test Report — M2

**Date:** 2026-08-31  
**Package version:** 0.0.3

## Full test suite

Command:

```bash
python -m pytest -q -ra
```

Result:

```text
83 passed, 3 skipped
```

Skipped by design because optional native dependencies are absent:

```text
SKIPPED tests/test_highspy_native_integration.py   optional native HiGHS integration
SKIPPED tests/test_osqp_native_integration.py      optional native OSQP integration
SKIPPED tests/test_pyscipopt_native_integration.py optional native SCIP integration
```

A skipped test is **not a pass**. The native code paths remain unverified against the installed upstream package until those dependencies can be installed.

## Compile verification

```bash
python -m compileall -q src tests benchmarks
```

Result: passed.

## Existing randomized mutation/hash checks

```bash
python benchmarks/m1_property_checks.py
```

Result:

```json
{
  "seeds": 250,
  "checks": 1000,
  "passed": 1000
}
```

These exercise update semantics, structural/data hashes, and mutation classification.

## OSQP translation property checks

```bash
python benchmarks/m2_osqp_translation_checks.py
```

Result:

```json
{
  "seeds": 200,
  "checks": 1200,
  "passed": 1200
}
```

The checks verify, for repeated same-structure convex QPs, that translated OSQP `P` and `A` retain identical CSC shapes/index arrays/indptr arrays while numerical vectors/values can change. This is a prerequisite for safe OSQP value-only update calls. It does **not** prove that the OSQP runtime accepted those updates because OSQP is not installed in the current runtime.

## Native adapter contract coverage without native dependencies

Active tests cover pure/helper behavior such as:

- HiGHS status normalization;
- HiGHS sparse LP matrix conversion;
- triangular Hessian conversion;
- OSQP QP data translation;
- OSQP status normalization;
- SCIP status normalization;
- optional backend availability/import-safety;
- registry behavior when native dependencies are missing.

## Session/reuse contract

Tests verify:

- registry persistence across Session solves;
- mutation history/revision behavior;
- reuse assessment by mutation type;
- `applied=False` when reuse is merely theoretically possible;
- `applied=True` only when a stateful backend execution explicitly reports applied reuse.

This prevents capability metadata from being mistaken for runtime evidence.

## Packaging smoke

Editable install with local build tooling and no dependency download:

```bash
python -m pip install --no-build-isolation --no-deps -e .
```

Result: passed.

Import/version smoke confirmed package version `0.0.3` and public `solve`/`Session` imports.

## Known test gap

Until an environment with the optional packages is available, these high-value tests remain pending:

- highspy LP/QP/MILP native end-to-end;
- OSQP repeated-QP update + automatic warm-start end-to-end;
- PySCIPOpt LP/MILP and IIS end-to-end;
- native solver version/ABI compatibility;
- native error/status edge cases;
- basis/MIP-start/reoptimization mechanisms;
- public benchmark corpora.
