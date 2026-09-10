"""Optional OR-Tools CP-SAT example. Install with: pip install 'solverpilot[cp]'"""
from solverpilot.cp import CPModel, ORToolsCPSATBackend, VERIFIED_ORTOOLS_VERSION

backend = ORToolsCPSATBackend()
if not backend.is_available():
    print(f"CP-SAT example skipped: install solverpilot[cp] (verified OR-Tools {VERIFIED_ORTOOLS_VERSION}).")
else:
    m = CPModel("cp-sat")
    x = m.bool_var("x")
    y = m.bool_var("y")
    m.add_exactly_one([x, y])
    m.maximize(3 * x + 2 * y)
    result = m.solve(backend)

    print("status:", result.status)
    print("objective:", result.objective)
    print("assignment:", result.assignment)

    assert result.validation.valid
    assert result.optimality_proven
    assert result.objective == 3
