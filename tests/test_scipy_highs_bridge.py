import numpy as np
import pytest

from solverpilot import LinearProblem, PublicStatus, QuadraticProblem, VariableDomain, execute
from solverpilot.backends import ScipyHighsBackend


def test_lp_end_to_end_optimal_and_validated():
    # min x + 2y, s.t. x+y >= 1, 0<=x,y<=1 => x=1,y=0, obj=1
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    r = execute(p, ScipyHighsBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.validation is not None and r.validation.valid
    assert abs(r.objective - 1.0) < 1e-8
    assert r.trace.backend == "scipy-highs-bridge"


def test_maximize_lp_preserves_original_objective_semantics():
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[2.0, 1.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[1.0],
        objective_sense="maximize",
        objective_offset=3.0,
    )
    r = execute(p, ScipyHighsBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert abs(r.objective - 5.0) < 1e-8


def test_binary_milp_end_to_end():
    # Maximize 3x + 2y, x+y <= 1, binary => x=1, obj=3
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[3.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[1.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
        objective_sense="maximize",
    )
    r = execute(p, ScipyHighsBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.validation is not None and r.validation.valid
    assert abs(r.objective - 3.0) < 1e-8
    assert np.max(np.abs(r.x - np.rint(r.x))) < 1e-8


def test_infeasible_status_has_no_fake_validation():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[2.0], constraint_upper=[np.inf],
    )
    r = execute(p, ScipyHighsBackend())
    assert r.status is PublicStatus.INFEASIBLE
    assert r.x is None
    assert r.validation is None


def test_unbounded_status():
    p = LinearProblem.from_data(
        A=np.zeros((0, 1)), c=[-1.0],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[], constraint_upper=[],
    )
    r = execute(p, ScipyHighsBackend())
    assert r.status is PublicStatus.UNBOUNDED
    assert r.x is None


def test_qp_is_rejected_by_capability_filter_before_backend_solve():
    p = QuadraticProblem.from_data(
        P=[[1.0]], A=np.zeros((0, 1)), q=[0.0],
        variable_lower=[-1.0], variable_upper=[1.0],
        constraint_lower=[], constraint_upper=[],
    )
    with pytest.raises(ValueError, match="incompatible"):
        execute(p, ScipyHighsBackend())
