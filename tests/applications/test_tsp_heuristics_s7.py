import random
import pytest
from solverpilot.applications.tsp import *


def asym(): return TSPInstance.from_distance_matrix([[0,5,1,9],[2,0,8,1],[9,1,0,2],[1,9,4,0]])
def sym(): return TSPInstance.from_distance_matrix([[0,9,2,7,3],[9,0,6,4,8],[2,6,0,5,1],[7,4,5,0,2],[3,8,1,2,0]])


def test_nearest_neighbor_valid():
    s=solve_tsp_nearest_neighbor(asym())
    assert validate_tsp_route(asym(),s).valid and not s.is_exact

def test_nearest_neighbor_tie_breaks_low_index():
    p=TSPInstance.from_distance_matrix([[0,1,1],[1,0,2],[1,2,0]])
    assert solve_tsp_nearest_neighbor(p).route[1]==1

def test_multistart_not_worse_than_start_zero():
    p=asym(); assert solve_tsp_multistart_nearest_neighbor(p).objective <= solve_tsp_nearest_neighbor(p).objective

def test_multistart_rejects_empty():
    with pytest.raises(TSPValidationError): solve_tsp_multistart_nearest_neighbor(asym(),start_indices=[])

def test_two_opt_symmetric_nonworsening_and_fast_delta():
    p=sym(); initial=(0,1,2,3,4,0); before=route_cost(p,initial); s=solve_tsp_two_opt(p,initial_route=initial)
    assert s.objective<=before+1e-12 and s.metadata["symmetric_fast_delta"] is True and validate_tsp_route(p,s).valid

def test_two_opt_asymmetric_nonworsening_and_recompute():
    p=asym(); initial=(0,1,2,3,0); before=route_cost(p,initial); s=solve_tsp_two_opt(p,initial_route=initial)
    assert s.objective<=before+1e-12 and s.metadata["symmetric_fast_delta"] is False and validate_tsp_route(p,s).valid

@pytest.mark.parametrize("strategy", ["best","first"])
def test_two_opt_strategies(strategy):
    s=solve_tsp_two_opt(sym(),strategy=strategy,max_iterations=20)
    assert validate_tsp_route(sym(),s).valid and s.metadata["strategy"]==strategy

def test_two_opt_rejects_invalid_route():
    with pytest.raises(TSPValidationError): solve_tsp_two_opt(sym(),initial_route=(0,1,1,3,4,0))

def test_two_opt_rejects_bad_strategy():
    with pytest.raises(TSPValidationError): solve_tsp_two_opt(sym(),strategy="magic")

def test_asymmetric_two_opt_regression_recomputes_internal_arc_directions():
    # Legacy symmetric-delta 2-opt reported 28 for its returned route while an
    # independent recomputation was 110 on this exact asymmetric fixture.
    m = [
        [0,27,15,1,5,41],
        [45,0,24,2,11,30],
        [1,8,0,17,21,3],
        [22,41,15,0,16,32],
        [47,44,36,10,0,20],
        [31,48,36,27,29,0],
    ]
    p=TSPInstance.from_distance_matrix(m)
    s=solve_tsp_two_opt(p,initial_route=(0,1,2,3,4,5,0),max_iterations=20)
    rep=validate_tsp_route(p,s)
    assert rep.valid and rep.objective_recomputed==s.objective
    assert s.metadata["symmetric_fast_delta"] is False
