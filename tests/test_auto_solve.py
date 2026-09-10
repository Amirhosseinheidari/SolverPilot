import numpy as np
import pytest

from solverpilot import LinearProblem, VariableDomain, solve
from solverpilot.validate import PublicStatus


def test_auto_solve_lp_produces_plan_and_timings():
    problem = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    result = solve(problem)
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.plan is not None
    assert result.plan.selected_backend == "scipy-highs-ds"
    assert result.trace.timings.inspect_s >= 0
    assert result.trace.timings.plan_s >= 0
    assert result.trace.timings.total_s >= result.trace.timings.solve_s


def test_auto_solve_milp_routes_to_milp_capable_backend():
    problem = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[-2.0, -1.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[1.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )
    result = solve(problem)
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.plan is not None
    assert result.plan.selected_backend == "scipy-highs-bridge"
    assert result.objective == pytest.approx(-2.0)


def test_explicit_backend_bypasses_plan_but_still_inspects_and_validates():
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    result = solve(problem, backend="scipy-highs-ipm")
    assert result.plan is None
    assert result.status is PublicStatus.VALID_OPTIMAL
    assert result.trace.backend == "scipy-highs-ipm"


def test_wall_time_budget_is_applied_only_when_backend_can_enforce_it():
    from solverpilot import SolveBudget
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    result = solve(problem, budget=SolveBudget(wall_time_s=1.0))
    assert result.status is PublicStatus.VALID_OPTIMAL


def test_unsupported_thread_budget_is_not_silently_ignored():
    from solverpilot import SolveBudget
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    with pytest.raises(NotImplementedError, match="thread budget"):
        solve(problem, budget=SolveBudget(threads=2))


def test_auto_trace_records_fingerprint_and_planner_evidence():
    problem = LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    result = solve(problem)
    assert result.trace.fingerprint is not None
    assert result.trace.fingerprint["problem_class"] == "lp"
    assert result.trace.planner_selected_backend == result.plan.selected_backend
    assert result.trace.planner_evidence_level == "capability_only"


def test_thread_budget_can_be_applied_to_backend_that_exposes_threads():
    from solverpilot import SolveBudget
    from solverpilot.backends import HighspyNativeBackend
    from solverpilot.runtime.auto import _apply_budget

    original = HighspyNativeBackend(threads=None)
    updated = _apply_budget(original, SolveBudget(threads=3, wall_time_s=2.5))
    assert updated is not original
    assert updated.threads == 3
    assert updated.time_limit_s == pytest.approx(2.5)
    assert original.threads is None
    assert original.time_limit_s is None
