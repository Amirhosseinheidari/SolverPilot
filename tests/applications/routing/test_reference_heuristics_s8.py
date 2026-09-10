import pytest
from solverpilot.applications.routing import *


def test_reference_exact_small(cvrp_small):
    r=solve_vrp_reference(cvrp_small)
    assert r.solution is not None and r.solution.objective==13
    assert r.solution.is_exact and r.solution.optimality_proven
    assert validate_vrp_solution(cvrp_small,r.solution).valid


def test_reference_vrptw(vrptw_small):
    r=solve_vrp_reference(vrptw_small)
    assert r.solution is not None and r.solution.objective==4
    assert r.solution.routes[0].customer_ids==('A','B')


def test_reference_infeasible_returns_none():
    nodes=[VRPNode('D'),VRPNode('A')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A',2)],[VRPVehicle('v','D',capacity=1)],[[0,1],[1,0]])
    r=solve_vrp_reference(inst)
    assert r.solution is None


def test_reference_limit_guard(cvrp_small):
    with pytest.raises(VRPReferenceLimitError): solve_vrp_reference(cvrp_small,max_customers=2)
    with pytest.raises(VRPReferenceLimitError): solve_vrp_reference(cvrp_small,max_evaluations=1)


def test_heuristic_feasible(cvrp_small):
    s=nearest_feasible_insertion(cvrp_small)
    assert validate_vrp_solution(cvrp_small,s).valid
    assert not s.is_exact and not s.optimality_proven


def test_heuristic_never_claims_optimality(vrptw_small):
    s=nearest_feasible_insertion(vrptw_small)
    assert not s.is_exact and not s.optimality_proven


def test_heuristic_partial_infeasible():
    nodes=[VRPNode('D'),VRPNode('A')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A',2)],[VRPVehicle('v','D',capacity=1)],[[0,1],[1,0]])
    s=nearest_feasible_insertion(inst)
    assert s.unassigned_customer_ids==('A',)
    assert not validate_vrp_solution(inst,s).valid


def test_local_search_does_not_worsen(cvrp_small):
    a=nearest_feasible_insertion(cvrp_small,improve=False)
    b=nearest_feasible_insertion(cvrp_small,improve=True)
    assert b.objective <= a.objective + 1e-9


def test_asymmetric_local_search_is_recomputed_not_symmetric_delta():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B'),VRPNode('C')]
    d=[[0,1,9,9],[9,0,1,9],[9,9,0,1],[1,9,9,0]]
    inst=VRPInstance(nodes,[VRPCustomer('A','A'),VRPCustomer('B','B'),VRPCustomer('C','C')],[VRPVehicle('v','D',capacity=10)],d)
    s=nearest_feasible_insertion(inst,improve=True)
    assert validate_vrp_solution(inst,s).valid
    assert s.objective==4


def test_clarke_wright_valid_classic_cvrp(cvrp_small):
    # fixture is symmetric, same depot, homogeneous capacity
    s=clarke_wright_savings(cvrp_small)
    assert validate_vrp_solution(cvrp_small,s).valid
    assert not s.is_exact and not s.optimality_proven


def test_clarke_wright_fails_closed_outside_scope(vrptw_small):
    with pytest.raises(VRPValidationError): clarke_wright_savings(vrptw_small)


def test_clarke_wright_rejects_asymmetric():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A'),VRPCustomer('B','B')],[VRPVehicle('v','D',capacity=2)],[[0,1,2],[4,0,1],[2,3,0]])
    with pytest.raises(VRPValidationError): clarke_wright_savings(inst)


def test_clarke_wright_does_not_merge_nonpositive_savings():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B')]
    # serving together costs 1+10+1=12, singleton routes cost 4; savings is negative
    d=[[0,1,1],[1,0,10],[1,10,0]]
    inst=VRPInstance(nodes,[VRPCustomer('A','A'),VRPCustomer('B','B')],[VRPVehicle('v1','D',capacity=2),VRPVehicle('v2','D',capacity=2)],d)
    s=clarke_wright_savings(inst)
    assert s.objective==4
    assert validate_vrp_solution(inst,s).valid
