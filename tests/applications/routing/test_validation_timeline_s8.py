import pytest
from solverpilot.applications.routing import *


def _codes(report): return {i.code for i in report.issues}


def test_valid_cvrp_solution(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B','C'])],13,method='fixture',is_exact=False,optimality_proven=False)
    r=validate_vrp_solution(cvrp_small,s)
    assert r.valid and r.objective_recomputed==13
    assert dict(r.route_loads)=={'v1':2,'v2':4}


def test_missing_customer_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B'])],10,method='x',is_exact=False,optimality_proven=False,unassigned_customer_ids=['C'])
    r=validate_vrp_solution(cvrp_small,s)
    assert not r.valid and 'unassigned_customer' in _codes(r)


def test_unassigned_declaration_must_match(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B'])],10,method='x',is_exact=False,optimality_proven=False)
    assert 'unassigned_declaration_mismatch' in _codes(validate_vrp_solution(cvrp_small,s))


def test_duplicate_customer_global_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A','B']),VRPRoute('v2',['B','C'])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'duplicate_customer_assignment' in _codes(validate_vrp_solution(cvrp_small,s))


def test_duplicate_customer_in_route_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A','A']),VRPRoute('v2',['B','C'])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'duplicate_customer_in_route' in _codes(validate_vrp_solution(cvrp_small,s))


def test_duplicate_vehicle_route_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v1',['B']),VRPRoute('v2',['C'])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'duplicate_vehicle_route' in _codes(validate_vrp_solution(cvrp_small,s))


def test_unknown_vehicle_detected(cvrp_small):
    s=VRPSolution([VRPRoute('z',['A']),VRPRoute('v2',['B','C'])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'unknown_vehicle' in _codes(validate_vrp_solution(cvrp_small,s))


def test_unknown_customer_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['Z']),VRPRoute('v2',['A','B','C'])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'unknown_customer' in _codes(validate_vrp_solution(cvrp_small,s))


def test_capacity_exceeded_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A','B','C']),VRPRoute('v2',[])],20,method='x',is_exact=False,optimality_proven=False)
    assert 'capacity_exceeded' in _codes(validate_vrp_solution(cvrp_small,s))


def test_objective_mismatch_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B','C'])],999,method='x',is_exact=False,optimality_proven=False)
    assert 'objective_mismatch' in _codes(validate_vrp_solution(cvrp_small,s))


def test_served_and_unassigned_detected(cvrp_small):
    s=VRPSolution([VRPRoute('v1',['A']),VRPRoute('v2',['B','C'])],13,method='x',is_exact=False,optimality_proven=False,unassigned_customer_ids=['A'])
    assert 'served_and_unassigned' in _codes(validate_vrp_solution(cvrp_small,s))


def test_timeline_waiting_and_service(vrptw_small):
    tl=build_route_timeline(vrptw_small,VRPRoute('v',['B','A']))
    # B arrives at 4, waits until 5, then A becomes late.
    assert tl.visits[0].waiting==1
    assert 'late_arrival:A' in tl.violations


def test_valid_timeline(vrptw_small):
    tl=build_route_timeline(vrptw_small,VRPRoute('v',['A','B']))
    assert tl.valid and tl.return_time==10 and tl.route_duration==10


def test_shift_late_return_detected():
    nodes=[VRPNode('D'),VRPNode('A')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A')],[VRPVehicle('v','D',capacity=1,shift_window=VRPTimeWindow(0,3))],[[0,1],[1,0]],travel_times=[[0,2],[2,0]])
    r=validate_vrp_solution(inst,VRPSolution([VRPRoute('v',['A'])],2,method='x',is_exact=False,optimality_proven=False))
    assert 'vehicle_shift_late_return' in _codes(r)


def test_max_duration_detected():
    nodes=[VRPNode('D'),VRPNode('A')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A')],[VRPVehicle('v','D',capacity=1,max_route_duration=3)],[[0,1],[1,0]],travel_times=[[0,2],[2,0]])
    r=validate_vrp_solution(inst,VRPSolution([VRPRoute('v',['A'])],2,method='x',is_exact=False,optimality_proven=False))
    assert 'max_route_duration_exceeded' in _codes(r)


def test_unused_vehicle_has_zero_time_and_distance(cvrp_small):
    route=VRPRoute('v1',[])
    assert route_distance(cvrp_small,route)==0


def test_timeline_requires_travel_matrix(cvrp_small):
    with pytest.raises(VRPValidationError): build_route_timeline(cvrp_small,VRPRoute('v1',['A']))


def test_validation_tolerances_reject_bool(cvrp_small):
    with pytest.raises(ValueError): validate_vrp_solution(cvrp_small,[],atol=True)


def test_solution_claim_invariant():
    with pytest.raises(VRPValidationError): VRPSolution([],0,method='x',is_exact=False,optimality_proven=True)
