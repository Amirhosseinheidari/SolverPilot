# M31 compatibility protocol

M31 is the release-candidate compatibility checkpoint after M30 API freeze. It separates **executed compatibility evidence** from **upstream wheel availability** and from **planned CI coverage**.

## Core matrix

The public compatibility target is CPython 3.10, 3.11, 3.12 and 3.13 across Linux, macOS and Windows. Each cell must install the package from a built distribution, run LP/MILP/QP smoke, verify conservative production planning, and run the full regression suite.

A cell is not called supported merely because Python or dependency wheels exist for it.

## Optional solver matrix

HiGHS, OSQP, PySCIPOpt, NLopt and the pinned CasADi verification bridge are exercised separately. Optional-backend failures do not silently degrade to a different backend in the compatibility test: the requested backend must become healthy.

## Fail-closed rules

- `pending_ci` is not a pass.
- upstream wheel availability is not execution evidence.
- Python-3.10 AST parsing is syntax evidence only.
- local DNS/network failure cannot be converted to a compatibility pass.
- absent legal/identity metadata cannot be fabricated.
- the 1.0 compatibility gate remains false until the intended matrix has actually run.

## CI implementation

`.github/workflows/m31-compatibility.yml` contains 47 jobs: 12 core OS/Python cells, 20 Ubuntu/Python optional-backend cells, and 15 Python-3.13 cross-OS optional-backend cells. It uses explicit `actions/setup-python` versions rather than runner-default Python.
