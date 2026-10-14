import time

import pytest
from solverpilot import LinearProblem
from solverpilot.runtime import batch


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
        variable_upper=[1.], constraint_lower=[0.], constraint_upper=[1.])


@pytest.mark.parametrize('mode', ['deadline', 'cancel'])
def test_isolated_candidate_is_rechecked_after_parent_validation(monkeypatch, mode):
    token = batch.CancellationToken()
    offset = [0.]
    monkeypatch.setattr(batch, 'monotonic', lambda: time.monotonic()+offset[0])
    validate = batch.validate_solution
    def slow_or_cancel(*args, **kwargs):
        result = validate(*args, **kwargs)
        assert result.valid
        if mode == 'deadline':
            offset[0] += 60.
        else:
            token.cancel()
        return result
    monkeypatch.setattr(batch, 'validate_solution', slow_or_cancel)
    p = problem()
    r, = batch.solve_batch([p], mode='isolated', backend='scipy-highs-ds',
                           timeout_s=30., cancellation=token)
    assert r.status == ('timeout' if mode == 'deadline' else 'cancelled')
    assert r.x is None and r.objective is None and not r.validation_valid
    assert not r.independently_verified_optimal
    assert r.problem_data_hash == p.data_hash and r.requested_time_s == 30.
    assert r.within_budget is (mode != 'deadline')


def test_isolated_completed_candidate_records_identity_and_budget():
    p = problem()
    r, = batch.solve_batch([p], mode='isolated', backend='scipy-highs-ds', timeout_s=30.)
    assert r.status == 'valid_optimal' and r.validation_valid
    assert r.problem_data_hash == p.data_hash
    assert r.within_budget and r.requested_time_s == 30.
