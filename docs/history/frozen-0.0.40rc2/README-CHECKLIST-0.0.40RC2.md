# README public-release checklist — 0.0.40rc2

The public README was checked against the requested end-user path. Every row below is backed by the current source tree and regression tests.

| # | Requirement | 0.0.40rc2 result |
| ---: | --- | --- |
| 1 | What SolverPilot is and what it does | PASS — opening definition + supported problem classes |
| 2 | Prerequisites and Python version | PASS — Python 3.12–3.14, NumPy >=2.2, SciPy >=1.15 |
| 3 | pip installation | PASS — PyPI command is shown with an explicit pre-publication caveat; source install is also shown |
| 4 | <2 minute Quick Start | PASS — minimal base-install LP |
| 5 | LP example | PASS — README + `examples/01_lp_basic.py` |
| 6 | MILP example | PASS — README + `examples/02_milp_binary.py` |
| 7 | QP example | PASS — README + `examples/03_qp_convex.py`; SLSQP is not mislabeled as proof-optimal |
| 8 | Backend selection/use | PASS — automatic routing, explicit backend ID, health CLI |
| 9 | status/objective/x/validation/diagnostics | PASS — field table + executable validation example |
| 10 | Infeasible problem | PASS — opt-in diagnostics example |
| 11 | Repeated solve/reoptimization | PASS — `Session` example + conservative reuse wording |
| 12 | Optional solver dependencies | PASS — extras mirror `pyproject.toml` |
| 13 | Unsupported capabilities | PASS — explicit section linked to `KNOWN-LIMITATIONS.md` |
| 14 | Complete examples under `examples/` | PASS — 7 executable examples + examples README |
| 15 | Separate documentation links | PASS — internal links are regression-tested |
| 16 | API Reference | PASS — `docs/api/PUBLIC-API-v1.md` + current 0.0.40RC2 snapshots |
| 17 | Verification that the structure really exists | PASS — `tests/test_public_docs_0.0.40rc2.py`, CI example execution, and sdist manifest enforcement |

Public examples are deliberately assertions-bearing scripts. They are not documentation snippets that can drift independently of the runtime.
