# Explicit cuOpt GPU LP integration (unreleased)

This development adapter is separate from the published SolverPilot 0.4 wheel.
It accepts continuous `LinearProblem` models and explicitly selects cuOpt PDLP.
It is absent from automatic/production routing. Installing a package or detecting
a GPU is not qualification, and one laptop does not establish a speed advantage
on another machine or a different problem family.

## Environment and use

Use Linux or Windows 11 with WSL2, a supported NVIDIA GPU, and a CUDA-compatible
host driver. For WSL, the Windows NVIDIA driver supplies GPU access; do not install
a Linux display driver inside the distribution. Start with a separate Python 3.12
environment. The optional GPU dependencies are large and must be installed in that
Linux environment, not a Windows Python environment.

```sh
python -m venv .venv
.venv/bin/python -m pip install --upgrade pip
.venv/bin/python -m pip install --extra-index-url https://pypi.nvidia.com '.[gpu,test,highs]'
```

```python
from solverpilot import LinearProblem, solve
from solverpilot.backends import CuOptBackend

p = LinearProblem.from_data(
    A=[[1., 1.], [1., -1.]], c=[-2., -1.],
    variable_lower=[0., 0.], variable_upper=[10., 10.],
    constraint_lower=[2., -2.], constraint_upper=[5., 1.],
)
r = solve(p, backend=CuOptBackend(time_limit_s=60, tolerance=1e-9))
print(r.objective, r.validation)
print(r.optimality_evidence.independently_verified_optimal)
```

`backend="cuopt-gpu"` selects the same adapter with default settings. `solve_any`
and compiled linear models can use the backend object. The historical top-level
API is unchanged; `CuOptBackend` lives in `solverpilot.backends`.

## Execution and trust

- Sparse CSR and binary64 arrays pass to an isolated Python worker. The caller
  does not import CUDA, cuOpt or cuDF. Worker startup, import and transport count
  against the adapter's wall-time limit; validation after the solve is separate.
- PDLP is selected explicitly. Concurrent CPU simplex, presolve and crossover are
  disabled for this initial path. FP64 is explicit. The response must report a CUDA device and PDLP.
- Original variable bounds, equality/ranged rows, maximization and objective
  offsets are preserved. Native primal objectives are independently recomputed;
  native duals are mapped back and checked by SolverPilot's numerical verifier.
- cuOpt 26.8 PDLP was observed returning zero reduced costs at active variable
  bounds. The adapter retains those raw values but reconstructs bound-dual
  candidates from the objective and row duals. Independent original-domain gap,
  residual-correction and complementarity checks still decide verification.
  For a bounds-only model, transport repeats one original bound as a redundant
  row because cuOpt rejects zero-row CSR; that row is removed on return.
- A native optimal status is not an independent proof. Inspect
  `optimality_evidence.independently_verified_optimal`. Free domains, insufficient
  accuracy or large residual corrections can leave a candidate unverified.
- Infeasibility remains a backend claim without a separately checked certificate.
  Dual infeasibility maps to `infeasible_or_unbounded`; it does not establish a
  feasible origin. Optional `certificate_recovery` retains its existing LP scope.
- No QP/MIP, automatic CPU fallback, GPU memory budget, warm start, persistent
  CUDA context, factorization reuse, or universally faster execution is claimed.
  Every call starts a new worker. Tiny problems can be dominated by startup.

## Repeatable qualification

CPU-only CI exercises protocol rejection, budget mapping, corrupted results,
objective/dual transformations, missing dependencies and the automatic-routing
exclusion. It does not execute GPU kernels.

Run actual GPU tests explicitly on the target machine:

```sh
SOLVERPILOT_TEST_CUOPT=1 python -m pytest tests/test_cuopt_native.py -ra
python tools/qualify_cuopt.py --output gpu-qualification.json
python -m pip freeze > gpu-environment.txt
```

The bounded comparison alternates execution order, uses reproducible sparse LPs
with known primal/dual optima, records model hashes and solver/package versions,
and reports complete wall time alongside native solve and validation timings.
Both feasibility and independent numerical optimality must pass. Results are
synthetic observations, not training/held-out evidence for a production policy.
Ordinary GitHub-hosted CI has no configured NVIDIA GPU; local GPU evidence must
remain clearly distinguished from CPU-only GitHub checks.

Official references:

- [cuOpt requirements](https://docs.nvidia.com/cuopt/user-guide/latest/system-requirements.html)
- [cuOpt solver settings](https://docs.nvidia.com/cuopt/user-guide/latest/convex-settings.html)
- [CUDA on WSL](https://docs.nvidia.com/cuda/wsl-user-guide/index.html)
- [CUDA minor-version compatibility and limitations](https://docs.nvidia.com/deploy/cuda-compatibility/minor-version-compatibility.html)
