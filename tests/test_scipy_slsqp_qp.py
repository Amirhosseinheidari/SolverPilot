import numpy as np
import pytest

from solverpilot import QuadraticProblem, solve
from solverpilot.backends import ScipySLSQPQPBackend
from solverpilot.runtime import execute
from solverpilot.validate import PublicStatus


def test_qp_bridge_unconstrained_box_optimum():
    # min (x-1)^2 + 2(y+0.5)^2 up to a constant.
    problem = QuadraticProblem.from_data(
        P=np.diag([2.0, 4.0]),
        A=np.zeros((0, 2)),
        q=[-2.0, 2.0],
        variable_lower=[-2.0, -2.0],
        variable_upper=[2.0, 2.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    result = execute(problem, ScipySLSQPQPBackend())
    assert result.status is PublicStatus.VALID_FEASIBLE
    assert result.validation is not None and result.validation.valid
    assert result.x == pytest.approx([1.0, -0.5], abs=1e-6)


def test_auto_qp_routes_to_development_qp_bridge_without_optimality_claim(scipy_registry):
    problem = QuadraticProblem.from_data(
        P=[[2.0]], A=[[1.0]], q=[-2.0],
        variable_lower=[0.0], variable_upper=[3.0],
        constraint_lower=[-np.inf], constraint_upper=[2.0],
    )
    result = solve(problem, registry=scipy_registry)
    assert result.plan is not None
    assert result.plan.selected_backend == "scipy-slsqp-qp-bridge"
    assert result.status is PublicStatus.VALID_FEASIBLE
    assert result.x == pytest.approx([1.0], abs=1e-6)
