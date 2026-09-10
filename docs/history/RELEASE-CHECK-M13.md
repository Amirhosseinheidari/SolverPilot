# RELEASE CHECK M13

**Version:** 0.0.14

## Source gates

- compileall `src/tests/benchmarks/tools`: PASS
- final JUnit: 218 pass / 3 skip / 0 fail / 0 error
- direct C-API stress: PASS
- repeated mixed health-probe regression: PASS

## Wheel gates

Wheel: `dist/optimind_core_codename-0.0.14-py3-none-any.whl`

SHA-256: `c9a29102e1d891fa6f034922368548841811b2953309767a6dc872c977324f01`

Installed with `--target` outside the source tree. Verified:

- imported module path comes from isolated install target;
- package version = distribution version = 0.0.14;
- direct HiGHS LP solve and explicit basis reuse;
- direct OSQP QP solve and persistent-workspace vector reuse;
- backend health three repeated rounds with bundled direct C APIs healthy;
- backend-health CLI has no `unhealthy` backends;
- benchmark no-site worker solves MPS and matches optimum reference;
- `optimind-bench list-datasets` and `doctor` execute from installed wheel.

## External gates still open

Public wheel URLs were attempted again. `highspy`, `osqp`, and `PySCIPOpt` all failed before download because `files.pythonhosted.org` could not be resolved from this runtime.

MIPLIB acquisition from the final wheel also failed with `URLError: Temporary failure in name resolution`.

Therefore M13 does **not** claim dedicated public-package integration or full MIPLIB/QPLIB execution.
