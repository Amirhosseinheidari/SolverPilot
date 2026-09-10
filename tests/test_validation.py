import numpy as np

from solverpilot import (
    CandidateSolution,
    LinearProblem,
    QuadraticProblem,
    ValidationTolerances,
    VariableDomain,
    validate_solution,
)


def test_valid_lp_solution_and_objective():
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[2.0, 1.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[1.0],
    )
    r = validate_solution(p, CandidateSolution(np.array([0.25, 0.75]), 1.25))
    assert r.valid
    assert r.objective_recomputed == 1.25
    assert r.objective_consistent is True


def test_constraint_violation_detected():
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[0.0, 0.0],
        variable_lower=[0.0, 0.0], variable_upper=[2.0, 2.0],
        constraint_lower=[1.0], constraint_upper=[1.0],
    )
    r = validate_solution(p, CandidateSolution(np.array([0.2, 0.2])))
    assert not r.valid
    assert r.max_constraint_violation > 0.5
    assert "constraint violation" in r.warnings


def test_bound_violation_detected():
    p = LinearProblem.from_data(
        A=[[0.0]], c=[0.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[0.0],
    )
    r = validate_solution(p, CandidateSolution(np.array([1.1])))
    assert not r.valid
    assert r.max_bound_violation > 0.09


def test_integrality_violation_detected():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[10.0],
        constraint_lower=[0.0], constraint_upper=[10.0],
        domains=[VariableDomain.INTEGER],
    )
    r = validate_solution(p, CandidateSolution(np.array([1.2])))
    assert not r.valid
    assert r.max_integrality_violation > 0.19


def test_objective_mismatch_detected():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[2.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    r = validate_solution(p, CandidateSolution(np.array([0.5]), objective_reported=99.0))
    assert not r.valid
    assert r.objective_consistent is False


def test_tolerance_is_respected():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[0.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[1.0], constraint_upper=[1.0],
    )
    r = validate_solution(
        p,
        CandidateSolution(np.array([1.0 + 5e-8])),
        tolerances=ValidationTolerances(feasibility=1e-7),
    )
    assert r.valid


def test_qp_objective_recomputed():
    p = QuadraticProblem.from_data(
        P=[[2.0]], A=[[1.0]], q=[3.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[0.0], constraint_upper=[2.0],
        objective_offset=4.0,
    )
    r = validate_solution(p, CandidateSolution(np.array([1.0]), objective_reported=8.0))
    assert r.valid
    assert abs(r.objective_recomputed - 8.0) < 1e-12


def test_nonfinite_solution_invalid():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    r = validate_solution(p, CandidateSolution(np.array([np.nan])))
    assert not r.valid


def test_empty_constraint_system_validates_bounds_only():
    p = LinearProblem.from_data(
        A=np.zeros((0, 2)), c=[1.0, -1.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[], constraint_upper=[],
    )
    r = validate_solution(p, CandidateSolution(np.array([0.2, 0.7]), objective_reported=-0.5))
    assert r.valid
    assert r.max_constraint_violation == 0.0


def test_negative_tolerance_rejected():
    import pytest
    with pytest.raises(ValueError, match="non-negative"):
        ValidationTolerances(feasibility=-1.0)
