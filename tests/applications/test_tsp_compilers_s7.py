import numpy as np
import pytest
from solverpilot.applications.tsp import *
from solverpilot.backends import ScipyHighsBackend
from solverpilot.cp import ReferenceCPBackend
from solverpilot.validate import PublicStatus


def p4(): return TSPInstance.from_distance_matrix([[0,3,9,2],[4,0,2,6],[7,1,0,3],[2,5,4,0]])
def int3(): return TSPInstance.from_distance_matrix([[0,4,1],[2,0,3],[5,1,0]])


def test_milp_compilation_dimensions():
    c=compile_tsp_milp(p4())
    assert len(c.arc_variables)==12 and len(c.order_variables)==3 and c.problem.n_variables==15

def test_milp_compilation_is_milp():
    c=compile_tsp_milp(p4()); assert c.problem.has_integer_variables

def test_milp_real_solve_matches_bruteforce():
    p=p4(); r=solve_tsp_milp(p,backend=ScipyHighsBackend())
    assert r.solution and r.validation and r.validation.valid
    assert r.solution.objective==solve_tsp_brute_force(p).objective
    assert r.core_result.status is PublicStatus.VALID_OPTIMAL

def test_milp_asymmetric_start_index():
    p=p4(); r=solve_tsp_milp(p,start_index=2)
    assert r.solution and r.solution.route[0]==2 and r.solution.route[-1]==2

def test_milp_decode_rejects_bad_shape():
    c=compile_tsp_milp(p4())
    with pytest.raises(TSPCompileError): decode_tsp_milp_solution(p4(),c,np.zeros(2))

def test_cp_compilation_has_circuit():
    c=compile_tsp_cp(int3())
    assert len(c.arc_variables)==6 and len(c.problem.constraints)==1

def test_cp_reference_matches_bruteforce():
    p=int3(); r=solve_tsp_cp(p,backend=ReferenceCPBackend(max_states=1000))
    assert r.solution and r.validation and r.validation.valid and r.core_result.optimality_proven
    assert r.solution.objective==solve_tsp_brute_force(p).objective

def test_cp_rejects_fractional_cost():
    p=TSPInstance.from_distance_matrix([[0,1.5],[2,0]])
    with pytest.raises(TSPCompileError): compile_tsp_cp(p)

def test_cp_rejects_nonexact_binary64_integer():
    x=float(1<<53)
    p=TSPInstance.from_distance_matrix([[0,x],[1,0]])
    with pytest.raises(TSPCompileError): compile_tsp_cp(p)

def test_cp_decode_rejects_empty_assignment():
    p=int3(); c=compile_tsp_cp(p)
    with pytest.raises(TSPCompileError): decode_tsp_cp_solution(p,c,{})

def test_compilers_bad_start():
    with pytest.raises(TSPValidationError): compile_tsp_milp(p4(),start_index=99)

def test_milp_solution_does_not_claim_independent_proof_without_evidence():
    r=solve_tsp_milp(p4())
    assert r.solution and r.solution.is_exact and r.solution.optimality_proven is False

def test_cp_solution_claim_matches_reference_proof():
    r=solve_tsp_cp(int3(),backend=ReferenceCPBackend(max_states=1000))
    assert r.solution and r.solution.optimality_proven is True

def test_compiled_problem_metadata_is_application_scoped():
    c=compile_tsp_milp(p4())
    assert c.problem.metadata["application"]=="tsp" and c.problem.metadata["formulation"]=="directed-mtz"

def test_milp_decoder_rejects_route_with_invalid_order_variables():
    p=p4(); c=compile_tsp_milp(p)
    x=np.zeros(c.problem.n_variables)
    for arc in [(0,1),(1,2),(2,3),(3,0)]: x[c.arc_variables[arc]]=1
    # order variables remain zero although their lower bound is one.
    with pytest.raises(TSPCompileError): decode_tsp_milp_solution(p,c,x)


def test_cp_decoder_rejects_fractional_raw_assignment():
    p=int3(); c=compile_tsp_cp(p)
    with pytest.raises(TSPCompileError): decode_tsp_cp_solution(p,c,{next(iter(c.arc_variables.values())):0.5})


def test_nonverification_cp_backend_does_not_upgrade_solver_proof_to_independent_proof():
    class SolverCertifiedOnly:
        verification_only=False
        def solve(self, problem, **kwargs):
            return ReferenceCPBackend(max_states=1000).solve(problem, **kwargs)
    r=solve_tsp_cp(int3(),backend=SolverCertifiedOnly())
    assert r.solution is not None
    assert r.solution.metadata["backend_optimality_proven"] is True
    assert r.solution.optimality_proven is False

def test_two_node_exact_compilers_match_reference():
    p=TSPInstance.from_distance_matrix([[0,7],[3,0]])
    expected=10.0
    assert solve_tsp_brute_force(p).objective==expected
    assert solve_tsp_milp(p).solution.objective==expected
    assert solve_tsp_cp(p,backend=ReferenceCPBackend(max_states=16)).solution.objective==expected


def test_fake_backend_cannot_self_declare_independent_reference_status():
    class PretendReference:
        verification_only=True
        def solve(self, problem, **kwargs):
            return ReferenceCPBackend(max_states=1000).solve(problem, **kwargs)
    r=solve_tsp_cp(int3(),backend=PretendReference())
    assert r.solution is not None and r.solution.optimality_proven is False
