import numpy as np
import pytest

from solverpilot import LinearProblem, ObjectiveSense, VariableDomain, execute
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.validate import PublicStatus


def _lp_problem(*, sense=ObjectiveSense.MINIMIZE):
    # min x + 2y, x+y >= 1, 0<=x,y<=1 -> x=1,y=0
    return LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
        objective_sense=sense,
    )


@pytest.mark.parametrize("method", ["highs-ds", "highs-ipm"])
def test_lp_backends_solve_min(method):
    result = execute(_lp_problem(), ScipyHighsLPBackend(method=method))
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.validation is not None and result.validation.valid
    assert result.objective == pytest.approx(1.0, abs=1e-8)
    assert result.x == pytest.approx([1.0, 0.0], abs=1e-7)


@pytest.mark.parametrize("method", ["highs-ds", "highs-ipm"])
def test_lp_backends_support_ranged_and_equality(method):
    # x = y and 1 <= x+y <= 1.5; minimize -x-y -> x=y=.75
    problem = LinearProblem.from_data(
        A=[[1.0, -1.0], [1.0, 1.0]],
        c=[-1.0, -1.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[2.0, 2.0],
        constraint_lower=[0.0, 1.0],
        constraint_upper=[0.0, 1.5],
    )
    result = execute(problem, ScipyHighsLPBackend(method=method))
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.objective == pytest.approx(-1.5, abs=1e-8)
    assert result.x == pytest.approx([0.75, 0.75], abs=1e-7)


def test_lp_backend_rejects_integer_model():
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[-1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[-np.inf], constraint_upper=[1.0],
        domains=[VariableDomain.BINARY],
    )
    backend = ScipyHighsLPBackend()
    with pytest.raises(ValueError, match="incompatible"):
        execute(problem, backend)


def test_lp_backend_rejects_bad_method():
    with pytest.raises(ValueError, match="method"):
        ScipyHighsLPBackend(method="highs")


def test_lp_backend_infeasible():
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[2.0], constraint_upper=[np.inf],
    )
    result = execute(problem, ScipyHighsLPBackend())
    assert result.status is PublicStatus.INFEASIBLE
    assert result.x is None
