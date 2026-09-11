# SolverPilot examples

Run examples from the repository root, for example:

```bash
python examples/01_lp_basic.py
python examples/08_semantic_model.py
python examples/12_cp_reference.py
```

## Base-runtime examples

| Example | Purpose |
| --- | --- |
| `01_lp_basic.py` | continuous LP |
| `02_milp_binary.py` | binary MILP |
| `03_qp_convex.py` | continuous convex QP |
| `04_choose_backend.py` | automatic vs explicit backend selection |
| `05_infeasible.py` | infeasibility and diagnostics |
| `06_validation_diagnostics.py` | `SolveResult` / validation fields |
| `07_reoptimization_session.py` | lightweight repeated-solve `Session` |
| `08_semantic_model.py` | semantic `Model` -> canonical LP compile/solve |
| `09_conic_model.py` | SOC representation and original-space validation |
| `12_cp_reference.py` | dependency-free CP reference backend |

## Optional-integration examples

| Example | Extra | Purpose |
| --- | --- | --- |
| `10_nlp_optional.py` | `solverpilot[nlp]` | smooth NLP local-optimal candidate |
| `11_minlp_optional.py` | `solverpilot[minlp]` | certified convex binary MINLP |
| `13_cp_sat_optional.py` | `solverpilot[cp]` | OR-Tools CP-SAT 9.15.6755 |
| `14_persistent_session_optional.py` | CasADi/bundled verification path | persistent native-patch routing |

Optional scripts exit successfully with an explanatory message when their verified dependency path is unavailable. Separate integration/conformance tests enforce behavior when those dependencies are present.

## Version 0.2

15: named production model, explicit soft penalty and quality evidence.
16: direct Clarabel PSD solve (optional clarabel extra).
17: bounded process batch scenarios (script main guard).

## Version 0.3

18: named sensitivity and independent parameter scenarios.
19: convex Huber loss and norm constraint (optional Clarabel).
20: streaming batches with bounded lookahead and persistent processes.
21: checked numerical optimum and opt-in JSON replay.
