import math
import pytest
from solverpilot.applications.tsp import *


def square4():
    return TSPInstance.from_distance_matrix([
        [0,1,2,1],[1,0,1,2],[2,1,0,1],[1,2,1,0]
    ], name="square4")


def test_distance_matrix_accepts_valid():
    dm=TSPDistanceMatrix([[0,1],[2,0]])
    assert dm.n_nodes==2 and dm.edge_cost(1,0)==2

def test_distance_matrix_rejects_too_small():
    with pytest.raises(TSPValidationError): TSPDistanceMatrix([[0]])

def test_distance_matrix_rejects_nonsquare():
    with pytest.raises(TSPValidationError): TSPDistanceMatrix([[0,1],[1]])

@pytest.mark.parametrize("bad", [float("nan"), float("inf"), -1.0])
def test_distance_matrix_rejects_bad_values(bad):
    with pytest.raises(TSPValidationError): TSPDistanceMatrix([[0,bad],[1,0]])

def test_node_requires_both_coordinates():
    with pytest.raises(TSPValidationError): TSPNode("x",1,None)

def test_instance_rejects_duplicate_ids():
    with pytest.raises(TSPValidationError): TSPInstance([TSPNode("a"),TSPNode("a")], [[0,1],[1,0]])

def test_from_coordinates_is_euclidean():
    p=TSPInstance.from_coordinates([(0,0),(3,4)],node_ids=["a","b"])
    assert p.distances.values==((0.0,5.0),(5.0,0.0))

def test_from_coordinates_rounding():
    p=TSPInstance.from_coordinates([(0,0),(1,1)],round_digits=2)
    assert p.distances.values[0][1]==1.41

def test_metric_analysis_square():
    a=analyze_distance_matrix(square4().distances)
    assert a.symmetric and a.zero_diagonal and a.triangle_inequality and a.metric

def test_metric_analysis_asymmetric_without_triangle():
    p=TSPInstance.from_distance_matrix([[0,1,5],[2,0,1],[1,2,0]])
    a=analyze_distance_matrix(p.distances,check_triangle=False)
    assert a.symmetric is False and a.triangle_inequality is None and a.metric is None

def test_route_cost():
    assert route_cost(square4(),(0,1,2,3,0))==4

def test_solution_bool_flags_are_strict():
    with pytest.raises(TSPValidationError): TSPSolution((0,1,0),2,method="x",is_exact="yes",optimality_proven=False)

def test_solution_cannot_claim_proof_for_heuristic():
    with pytest.raises(TSPValidationError): TSPSolution((0,1,0),2,method="x",is_exact=False,optimality_proven=True)

def test_distance_matrix_wraps_nonnumeric_error():
    with pytest.raises(TSPValidationError):
        TSPDistanceMatrix([[0, "not-a-number"], [1, 0]])


def test_node_wraps_bad_coordinate_error():
    with pytest.raises(TSPValidationError):
        TSPNode("x", "bad", 1)
