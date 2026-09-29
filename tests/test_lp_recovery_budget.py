"""The optional recovery budget includes preparation and the final wrapper check."""
import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.validate import lp_dual, optimality


@pytest.fixture
def problem():
    return LinearProblem.from_data(A=[[1.]], c=[-1.], variable_lower=[0.],
        variable_upper=[np.inf], constraint_lower=[-np.inf], constraint_upper=[3.])


@pytest.mark.parametrize('stage', ['dual', 'basis'])
def test_exhausted_preparation_never_runs_checker(problem, monkeypatch, stage):
    clock = [0.]
    monkeypatch.setattr(lp_dual, 'perf_counter', lambda: clock[0])
    name = 'prepare_lp_dual' if stage == 'dual' else 'validate_lp_basis'
    original = getattr(lp_dual, name)
    def delayed(*args):
        result = original(*args)
        clock[0] += .03
        return result
    monkeypatch.setattr(lp_dual, name, delayed)
    def unexpected(*args, **kwargs):
        pytest.fail('checker must not start after the preparation budget expires')
    monkeypatch.setattr(optimality, 'verify_optimality', unexpected)
    check = lp_dual.recover_lp_optimality(problem, [3.], [1., 0.], time_limit_s=.01,
        basis={'basic_columns': [0], 'nonbasic_rows': [0]})
    assert not check.verified and 'time limit' in check.reason
    assert check.recovery_diagnostics['wrapper']['elapsed_s'] == .03


@pytest.mark.parametrize('finish,verified', [(.009, True), (.01, False), (.02, False)])
def test_remaining_budget_and_late_positive_result(problem, monkeypatch, finish, verified):
    clock = [0.]
    monkeypatch.setattr(lp_dual, 'perf_counter', lambda: clock[0])
    original = lp_dual.prepare_lp_dual
    def prepare(*args):
        clock[0] = .004
        return original(*args)
    monkeypatch.setattr(lp_dual, 'prepare_lp_dual', prepare)
    def checked(*args, **kwargs):
        assert kwargs['recovery_options']['time_limit_s'] == pytest.approx(.006)
        clock[0] = finish
        return optimality.OptimalityCheck(True, True, True, 0., 0., 0., 'valid', -3.)
    monkeypatch.setattr(optimality, 'verify_optimality', checked)
    check = lp_dual.recover_lp_optimality(problem, [3.], [1., 0.], time_limit_s=.01)
    assert check.verified is verified
    assert check.dual_bound == -3. and check.primal_valid


@pytest.mark.parametrize('limit', [True, 0., -1., np.inf, np.nan])
def test_invalid_budget_rejected_before_preparation(problem, monkeypatch, limit):
    def unexpected(*args):
        pytest.fail('invalid budget must be rejected before preparation')
    monkeypatch.setattr(lp_dual, 'prepare_lp_dual', unexpected)
    with pytest.raises(ValueError, match='time limit'):
        lp_dual.recover_lp_optimality(problem, [3.], [1., 0.], time_limit_s=limit)


def test_explicit_unlimited_budget_keeps_existing_semantics(problem):
    check = lp_dual.recover_lp_optimality(problem, [3.], [1., 0.], time_limit_s=None)
    assert check.verified
