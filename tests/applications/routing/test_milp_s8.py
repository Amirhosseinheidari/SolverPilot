import numpy as np
import pytest
from solverpilot.validate import PublicStatus
from solverpilot.applications.routing import *
from solverpilot.applications.routing.compile_milp import SOURCE,SINK


def test_compile_has_binary_arcs_and_visits(cvrp_small):
    c=compile_vrp_milp(cvrp_small)
    assert c.problem.has_integer_variables
    assert len(c.visit_variables)==6
    assert ('v1',SOURCE,SINK) in c.arc_variables
    assert not c.time_variables


def test_compile_vrptw_has_time_variables(vrptw_small):
    c=compile_vrp_milp(vrptw_small)
    assert len(c.time_variables)==2
    assert c.big_m_time is not None and c.big_m_time>0


def test_milp_matches_reference_cvrp(cvrp_small):
    ref=solve_vrp_reference(cvrp_small).solution
    result=solve_vrp_milp(cvrp_small)
    assert result.solution is not None and result.validation.valid
    assert result.solution.objective==pytest.approx(ref.objective)
    assert result.core_result.status==PublicStatus.VALID_OPTIMAL
    assert result.solution.is_exact
    assert not result.solution.optimality_proven  # backend certificate is not an independent proof here


def test_milp_matches_reference_vrptw(vrptw_small):
    ref=solve_vrp_reference(vrptw_small).solution
    result=solve_vrp_milp(vrptw_small)
    assert result.solution is not None and result.validation.valid
    assert result.solution.objective==pytest.approx(ref.objective)


def test_milp_infeasible_has_no_solution():
    nodes=[VRPNode('D'),VRPNode('A')]
    inst=VRPInstance(nodes,[VRPCustomer('A','A',2)],[VRPVehicle('v','D',capacity=1)],[[0,1],[1,0]])
    result=solve_vrp_milp(inst)
    assert result.solution is None


def test_decode_rejects_wrong_shape(cvrp_small):
    c=compile_vrp_milp(cvrp_small)
    with pytest.raises(VRPCompileError): decode_vrp_milp_solution(cvrp_small,c,np.zeros(2))


def test_decode_rejects_fractional_candidate(cvrp_small):
    c=compile_vrp_milp(cvrp_small)
    x=np.zeros(c.problem.n_variables)
    for idx in c.arc_variables.values(): x[idx]=0.5
    with pytest.raises(VRPCompileError): decode_vrp_milp_solution(cvrp_small,c,x)


def test_heterogeneous_capacity_reference_equals_milp():
    nodes=[VRPNode('D'),VRPNode('A'),VRPNode('B')]
    d=[[0,1,4],[1,0,2],[4,2,0]]
    inst=VRPInstance(nodes,[VRPCustomer('A','A',3),VRPCustomer('B','B',1)],[VRPVehicle('small','D',capacity=1),VRPVehicle('large','D',capacity=3)],d)
    ref=solve_vrp_reference(inst).solution
    m=solve_vrp_milp(inst).solution
    assert ref and m and m.objective==pytest.approx(ref.objective)
    assert validate_vrp_solution(inst,m).valid


def test_distinct_start_end_nodes_supported():
    nodes=[VRPNode('S'),VRPNode('T'),VRPNode('A')]
    d=[[0,5,1],[5,0,1],[1,1,0]]
    inst=VRPInstance(nodes,[VRPCustomer('A','A',1)],[VRPVehicle('v','S','T',capacity=1)],d)
    m=solve_vrp_milp(inst).solution
    assert m and m.objective==2
