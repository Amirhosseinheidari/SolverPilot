# M11 Release Check — 0.0.12

## Source gate
- pytest: 204 total / 201 passed / 3 skipped / 0 failed / 0 errors.
- `compileall`: passed for `src`, `tests`, `benchmarks`, `tools`.
- numeric/property regressions: passed as recorded in `TEST-REPORT-M11.md`.

## Wheel
Built with:

```bash
python -m pip wheel . --no-deps --no-build-isolation -w dist
```

Artifact:
- `dist/optimind_core_codename-0.0.12-py3-none-any.whl`
- SHA-256: `5cb8a7f6c5a92bcdd7f37dd6b626ee8b634c78f87a1366d76bc97817223f6216`

## Installed-wheel smoke
The wheel was installed under a separate prefix, with the source tree excluded from imports. The installed `optimind-bench` command successfully executed:
- `list-datasets`;
- `doctor`;
- local `file://` acquisition with expected SHA-256;
- `verify-dataset`;
- process-isolated `run` using DS/IPM/`@auto`;
- `summarize` with `worker_solve_wall_s`;
- `evaluate-policy`;
- `emit-slurm`;
- `bundle`.

Installed version reported `0.0.12`; the tiny wheel-smoke benchmark produced 6/6 solved/validated rows and the generated dataset verification returned `ok=true`.

Audit files are under `benchmarks/results/m11-wheel-smoke/`.

## Network/public-native boundary
No dedicated highspy / OSQP / PySCIPOpt wheel was downloaded in M11. External name resolution remained unavailable. Their integration skips remain visible and are not counted as passes.

## Distribution decision
M11 is suitable as a research/HPC benchmark-handoff artifact. It is not a public production release and makes no MIPLIB/QPLIB performance claim.
