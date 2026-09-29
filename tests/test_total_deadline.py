import numpy as np
import pytest
from solverpilot import LinearProblem, solve, solve_production, SolveBudget
from solverpilot.runtime.deadline import solve_with_deadline


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],variable_upper=[2.],
        constraint_lower=[1.],constraint_upper=[np.inf])


def test_isolated_call_keeps_verified_evidence_and_identity():
    p = problem()
    r = solve_with_deadline(p, timeout_s=20., backend='scipy-highs-ds')
    assert r.validation_valid and r.independently_verified_optimal
    assert r.problem_data_hash == p.data_hash and r.objective == pytest.approx(1.)


def test_deadline_startup_failure_cannot_return_an_incumbent():
    r = solve_with_deadline(problem(), timeout_s=.00001)
    assert r.status == 'timeout' and r.x is None and not r.independently_verified_optimal
    with pytest.raises(ValueError): solve_with_deadline(problem(), timeout_s=1, backend='ortools-pdlp')


def test_budget_is_recorded_for_full_default_and_production_calls():
    p = problem()
    for r in [solve(p, budget=SolveBudget(wall_time_s=10)),
              solve_production(p, budget=SolveBudget(wall_time_s=10))[0]]:
        b = r.raw_statistics['call_budget']
        assert b['requested_s'] == 10 and b['within_budget']
        assert b['elapsed_s'] == r.trace.timings.total_s


def test_exhausted_setup_does_not_start_a_solver():
    with pytest.raises(TimeoutError): solve(problem(), budget=SolveBudget(wall_time_s=1e-15))
