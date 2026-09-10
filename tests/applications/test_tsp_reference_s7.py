import random
import pytest
from solverpilot.applications.tsp import *


def p4():
    return TSPInstance.from_distance_matrix([[0,3,9,2],[4,0,2,6],[7,1,0,3],[2,5,4,0]])


def test_bruteforce_known():
    s=solve_tsp_brute_force(p4())
    assert s.objective==10 and validate_tsp_route(p4(),s).valid

def test_held_karp_matches_brute_known():
    b=solve_tsp_brute_force(p4()); h=solve_tsp_held_karp(p4())
    assert h.objective==b.objective and h.optimality_proven

@pytest.mark.parametrize("seed", range(5))
def test_held_karp_matches_bruteforce_random(seed):
    r=random.Random(seed); n=6
    m=[[0 if i==j else r.randint(1,20) for j in range(n)] for i in range(n)]
    p=TSPInstance.from_distance_matrix(m)
    assert solve_tsp_held_karp(p,max_nodes=6).objective==solve_tsp_brute_force(p,max_nodes=6).objective

def test_exact_start_index_preserved():
    for solver in (solve_tsp_brute_force,solve_tsp_held_karp):
        s=solver(p4(),start_index=2,max_nodes=8)
        assert s.route[0]==2 and s.route[-1]==2

def test_bruteforce_limit():
    with pytest.raises(TSPReferenceLimitError): solve_tsp_brute_force(p4(),max_nodes=3)

def test_held_limit():
    with pytest.raises(TSPReferenceLimitError): solve_tsp_held_karp(p4(),max_nodes=3)

def test_exact_bad_start():
    with pytest.raises(TSPValidationError): solve_tsp_held_karp(p4(),start_index=9)

def test_validation_detects_missing_node():
    rep=validate_tsp_route(p4(),(0,1,2,1,0))
    assert not rep.valid and any("missing" in e for e in rep.errors)

def test_validation_detects_cost_mismatch():
    rep=validate_tsp_route(p4(),(0,1,2,3,0),reported_objective=999)
    assert not rep.valid and rep.objective_recomputed is not None

def test_validation_rejects_negative_tolerance():
    with pytest.raises(ValueError):
        validate_tsp_route(p4(), (0,1,2,3,0), atol=-1)


def test_validation_rejects_boolean_reported_objective():
    rep=validate_tsp_route(p4(), (0,1,2,3,0), reported_objective=True)
    assert not rep.valid and any("bool" in e for e in rep.errors)
