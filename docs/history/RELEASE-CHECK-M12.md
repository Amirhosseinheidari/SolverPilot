# M12 Release Check

- Source version: `0.0.13`.
- `compileall src tests benchmarks tools`: PASS.
- Wheel build: PASS.
- Wheel filename: `optimind_core_codename-0.0.13-py3-none-any.whl`.
- Isolated `--target` install: PASS.
- Installed module path was outside source tree.
- Installed `optimind.__version__`: `0.0.13`.
- Installed distribution metadata version: `0.0.13`.
- LP solve + canonical validation from installed wheel: PASS.
- infeasibility diagnosis from installed wheel: PASS.
- backend health CLI from installed wheel: PASS.
- benchmark CLI `list-datasets`, `doctor`, `run`, `summarize`, `evaluate-policy`, `emit-slurm`, `bundle`: PASS.
- wheel CLI fixed-SBS policy reconstruction ratio: 1.0.
- generated audit bundle `zipfile.testzip()`: PASS.

Public native health from installed wheel:
- `highspy-native`: `unavailable`.
- `osqp-native`: `unavailable`.
- `pyscipopt-native`: `unavailable`.

CasADi verification bridges report bridge package version 3.7.2 and are not included in the production default registry.
