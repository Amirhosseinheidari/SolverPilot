import importlib.util
import solverpilot
from solverpilot.applications import tsp
from solverpilot.applications.tsp import *


def test_top_level_api_freeze_is_78():
    assert len(solverpilot.__all__)==78

def test_tsp_not_in_top_level_api():
    assert "TSPInstance" not in solverpilot.__all__ and "solve_tsp_brute_force" not in solverpilot.__all__

def test_application_import_surface():
    assert TSPInstance is tsp.TSPInstance and callable(solve_tsp_milp)

def test_matrix_and_coordinates_same_square_objective():
    a=TSPInstance.from_coordinates([(0,0),(1,0),(1,1),(0,1)])
    b=TSPInstance.from_distance_matrix([[0,1,2**0.5,1],[1,0,1,2**0.5],[2**0.5,1,0,1],[1,2**0.5,1,0]])
    assert solve_tsp_held_karp(a).objective==solve_tsp_held_karp(b).objective==4.0

def test_cross_exact_methods_agree():
    p=TSPInstance.from_distance_matrix([[0,6,2,5],[4,0,3,8],[7,1,0,2],[3,9,4,0]])
    values=[solve_tsp_brute_force(p).objective,solve_tsp_held_karp(p).objective,solve_tsp_milp(p).solution.objective]
    assert values.count(values[0])==len(values)

def test_ortools_availability_is_not_assumed():
    # S7 CP formulation must be usable without OR-Tools; optional runtime qualification is separate.
    p=TSPInstance.from_distance_matrix([[0,1,2],[1,0,3],[2,3,0]])
    assert compile_tsp_cp(p).problem is not None
