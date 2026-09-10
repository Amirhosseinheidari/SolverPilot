from solverpilot.applications.routing import *


def test_capacity_diagnostic(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A','B','C']),VRPRoute('v2',[])],20,method='x',is_exact=False,optimality_proven=False)
    actions=diagnose_vrp_validation(validate_vrp_solution(cvrp_small,s))
    assert any(a.code=='capacity_exceeded' and a.severity=='blocking' for a in actions)


def test_objective_diagnostic(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B','C'])],999,method='x',is_exact=False,optimality_proven=False)
    actions=diagnose_vrp_validation(validate_vrp_solution(cvrp_small,s))
    assert any(a.code=='objective_mismatch' and 'recompute' in a.suggestion.lower() for a in actions)


def test_time_window_diagnostic(vrptw_small):
    s=VRPSolution([VRPRoute('v',['B','A'])],4,method='x',is_exact=False,optimality_proven=False)
    actions=diagnose_vrp_validation(validate_vrp_solution(vrptw_small,s))
    assert any(a.code=='time_window_violation' and 'resequence' in a.suggestion.lower() for a in actions)


def test_valid_solution_has_no_actions(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B','C'])],13,method='x',is_exact=False,optimality_proven=False)
    assert diagnose_vrp_validation(validate_vrp_solution(cvrp_small,s))==()


def test_waiting_is_visible_but_not_a_feasibility_error(vrptw_small):
    # B first waits one minute. The route later violates A, so inspect waiting action independently.
    s=VRPSolution([VRPRoute('v',['B','A'])],5,method='x',is_exact=False,optimality_proven=False)
    actions=diagnose_vrp_solution(vrptw_small,s,waiting_warning_threshold=0.5)
    assert any(a.code=='waiting_time_detected' and a.target=='B' and a.severity=='warning' for a in actions)


def test_waiting_threshold_rejects_bool(vrptw_small):
    s=VRPSolution([VRPRoute('v',['A','B'])],4,method='x',is_exact=False,optimality_proven=False)
    import pytest
    with pytest.raises(ValueError): diagnose_vrp_solution(vrptw_small,s,waiting_warning_threshold=True)


def test_waiting_threshold_rejects_nonfinite(vrptw_small):
    import pytest
    s=VRPSolution([VRPRoute('v',['A','B'])],4,method='x',is_exact=False,optimality_proven=False)
    with pytest.raises(ValueError): diagnose_vrp_solution(vrptw_small,s,waiting_warning_threshold=float('nan'))
