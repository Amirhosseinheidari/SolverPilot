import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, VariableDomain
from solverpilot.backends import BackendRegistry, ScipyHighsBackend, ScipyHighsLPBackend
from solverpilot.plan import NoCompatibleBackendError, PlannerContext, SolveBudget, plan_solve


def lp():
    return LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )


def milp():
    return LinearProblem.from_data(
        A=[[1.0]], c=[-1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
        domains=[VariableDomain.BINARY],
    )


def qp():
    return QuadraticProblem.from_data(
        P=[[2.0]], A=[[1.0]], q=[-1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[-np.inf], constraint_upper=[2.0],
    )


def registry():
    r = BackendRegistry()
    r.register(ScipyHighsLPBackend("highs-ds"))
    r.register(ScipyHighsLPBackend("highs-ipm"))
    r.register(ScipyHighsBackend())
    return r


def test_planner_filters_by_capability():
    plan = plan_solve(milp(), registry())
    assert plan.selected_backend == "scipy-highs-bridge"
    assert [c.backend for c in plan.candidates] == ["scipy-highs-bridge"]
    assert plan.evidence_level == "capability_only"


def test_lp_tie_uses_registration_order_and_says_so():
    plan = plan_solve(lp(), registry())
    assert plan.selected_backend == "scipy-highs-ds"
    assert len(plan.candidates) == 3
    assert any("registration order" in x for x in plan.rationale)


def test_previous_backend_is_only_small_continuity_tiebreak():
    plan = plan_solve(
        lp(), registry(), context=PlannerContext(previous_backend="scipy-highs-ipm")
    )
    assert plan.selected_backend == "scipy-highs-ipm"
    assert plan.candidates[0].score == pytest.approx(100.1)


def test_no_qp_backend_fails_cleanly():
    with pytest.raises(NoCompatibleBackendError, match="convex_qp"):
        plan_solve(qp(), registry())


def test_budget_validation():
    with pytest.raises(ValueError):
        SolveBudget(wall_time_s=0)


def test_prove_optimal_intent_rejects_candidate_only_qp_backends():
    import numpy as np
    import pytest
    from solverpilot import QuadraticProblem
    from solverpilot.plan import NoCompatibleBackendError, SolveIntent
    from solverpilot.runtime import default_registry
    from solverpilot.plan import plan_solve

    p = QuadraticProblem.from_data(
        P=[[2.0]], A=np.zeros((0, 1)), q=[-2.0],
        variable_lower=[0.0], variable_upper=[3.0],
        constraint_lower=[], constraint_upper=[],
    )
    # In the current test environment highspy/OSQP are absent. The remaining QP
    # backends (SciPy/NLopt SLSQP) explicitly lack an optimality certificate.
    with pytest.raises(NoCompatibleBackendError, match="rejected by solve intent"):
        plan_solve(p, default_registry(), intent=SolveIntent.PROVE_OPTIMAL)


def test_prove_optimal_intent_still_allows_highs_lp_backend():
    from solverpilot import LinearProblem
    from solverpilot.plan import SolveIntent
    from solverpilot.runtime import default_registry
    from solverpilot.plan import plan_solve

    p = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[1.0], constraint_upper=[float("inf")],
    )
    plan = plan_solve(p, default_registry(), intent=SolveIntent.PROVE_OPTIMAL)
    assert plan.selected_backend == "scipy-highs-ds"
    assert any("hard intent gate" in r for r in plan.rationale)
