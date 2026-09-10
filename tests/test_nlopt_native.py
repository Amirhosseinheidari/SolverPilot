import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, solve
from solverpilot.backends import NLoptNativeBackend
from solverpilot.runtime import execute
from solverpilot.validate import PublicStatus


pytestmark = pytest.mark.skipif(
    not NLoptNativeBackend().is_available(), reason="nlopt not installed"
)


def test_nlopt_lp_min_and_max_are_validated_candidates():
    pmin = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    rmin = execute(pmin, NLoptNativeBackend())
    assert rmin.status is PublicStatus.VALID_FEASIBLE
    assert rmin.validation is not None and rmin.validation.valid
    assert rmin.objective == pytest.approx(1.0, abs=1e-6)

    from solverpilot.problem import ObjectiveSense
    pmax = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[-np.inf], constraint_upper=[1.25],
        objective_sense=ObjectiveSense.MAXIMIZE,
        objective_offset=3.0,
    )
    rmax = execute(pmax, NLoptNativeBackend())
    assert rmax.status is PublicStatus.VALID_FEASIBLE
    assert rmax.validation is not None and rmax.validation.valid
    assert rmax.objective == pytest.approx(4.25, abs=1e-6)


def test_nlopt_qp_equality_known_kkt_optimum():
    # min 0.5*x'P*x + q'x, x0+x1=1. Choose x*=[.25,.75]
    # and lambda=.4 so q=-P*x*-A.T*lambda.
    P = np.diag([2.0, 4.0])
    x_star = np.array([0.25, 0.75])
    lam = 0.4
    A = np.array([[1.0, 1.0]])
    q = -(P @ x_star) - A.T[:, 0] * lam
    problem = QuadraticProblem.from_data(
        P=P, A=A, q=q,
        variable_lower=[-2.0, -2.0], variable_upper=[2.0, 2.0],
        constraint_lower=[1.0], constraint_upper=[1.0],
    )
    result = execute(problem, NLoptNativeBackend())
    assert result.status is PublicStatus.VALID_FEASIBLE
    assert result.validation is not None and result.validation.valid
    assert result.x == pytest.approx(x_star, abs=3e-6)


def test_nlopt_qp_active_upper_inequality_known_kkt_optimum():
    P = np.eye(2) * 2.0
    x_star = np.array([0.5, -0.25])
    a = np.array([1.0, 2.0])
    multiplier = 0.7
    q = -(P @ x_star) - multiplier * a
    b = float(a @ x_star)
    problem = QuadraticProblem.from_data(
        P=P, A=[a], q=q,
        variable_lower=[-3.0, -3.0], variable_upper=[3.0, 3.0],
        constraint_lower=[-np.inf], constraint_upper=[b],
    )
    result = execute(problem, NLoptNativeBackend())
    assert result.status is PublicStatus.VALID_FEASIBLE
    assert result.validation is not None and result.validation.valid
    assert result.x == pytest.approx(x_star, abs=5e-6)


def test_nlopt_rejects_milp():
    from solverpilot import VariableDomain
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
        domains=[VariableDomain.BINARY],
    )
    with pytest.raises(ValueError, match="continuous LP"):
        NLoptNativeBackend().solve(p)


def test_default_planner_does_not_prefer_nlopt_over_native_highs_for_lp():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    r = solve(p)
    assert r.plan is not None
    assert r.plan.selected_backend == "scipy-highs-ds"


def test_nlopt_does_not_claim_infeasibility_without_certificate():
    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[2.0], constraint_upper=[np.inf],
    )
    r = execute(p, NLoptNativeBackend(max_evals=100))
    # A local optimizer is not an infeasibility prover. Its candidate is independently
    # rejected rather than promoted to an infeasible certificate.
    assert r.status is PublicStatus.INVALID_SOLUTION
    assert r.validation is not None and not r.validation.valid
    assert r.backend_status != "infeasible"
