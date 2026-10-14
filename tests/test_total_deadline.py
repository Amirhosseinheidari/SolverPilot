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


def test_precancelled_deadline_returns_an_owned_cancelled_result_without_workers(monkeypatch):
    from solverpilot.runtime.batch import CancellationToken
    import solverpilot.runtime.deadline as deadline
    token = CancellationToken()
    token.cancel()
    monkeypatch.setattr(deadline, 'BatchExecutor', lambda **k: pytest.fail('worker started'))
    p = problem()
    r = solve_with_deadline(p, timeout_s=1., cancellation=token)
    assert r.status == 'cancelled' and r.x is None and not r.validation_valid
    assert not r.independently_verified_optimal and r.execution_id
    assert r.problem_data_hash == p.data_hash and r.requested_time_s == 1.


def test_cancellation_during_submission_does_not_leak_stopiteration(monkeypatch):
    from solverpilot.runtime.batch import CancellationToken
    import solverpilot.runtime.deadline as deadline
    token = CancellationToken()
    class EmptyExecutor:
        def __init__(self, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, *args):
            pass
        def iter(self, problems, *, cancellation):
            cancellation.cancel()
            return iter(())
    monkeypatch.setattr(deadline, 'BatchExecutor', EmptyExecutor)
    assert solve_with_deadline(problem(), timeout_s=1., cancellation=token).status == 'cancelled'
