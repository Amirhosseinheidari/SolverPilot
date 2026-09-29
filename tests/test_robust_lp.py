from dataclasses import replace

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.experimental.robust_lp import decide_robust_lp, solve_robust_lp
from solverpilot.experimental.learned_lp import LPSelector, lp_features


def case():
    p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                               constraint_lower=[1.], constraint_upper=[np.inf])
    features = lp_features(p)
    m = LPSelector(('a', 'b'), 'b', 'env', 'a'*64, 2., 0, 100., 'a', 'b',
                   features, features, ('old',), ('old-group',), ('b'*64,), 'c'*64)
    return p, m


@pytest.mark.parametrize('reason', ['environment', 'support', 'availability'])
def test_abstentions_use_production_not_training_baseline(reason):
    p, m = case()
    env, available = 'env', ('a', 'b')
    if reason == 'environment': env = 'different'
    if reason == 'support': m = replace(m, lower=(0.,)*8, upper=(0.,)*8)
    if reason == 'availability': available = ('b',)
    d = decide_robust_lp(p, m, environment_id=env, available=available)
    assert d.candidate == 'production' and d.reason == 'production_fallback'


def test_explicit_route_keeps_independent_checks_and_no_auto_enable():
    p, m = case()
    r = solve_robust_lp(p, m, environment_id='env', backends={'a': ScipyHighsLPBackend()})
    assert r.objective == pytest.approx(1.)
    assert r.optimality_evidence.independently_verified_optimal
    assert r.raw_statistics['experimental_lp_route']['candidate'] == 'a'
    assert r.raw_statistics['experimental_lp_route']['automatic_production_routing_enabled'] is False
    r = solve_robust_lp(p, m, environment_id='different', backends={})
    assert r.objective == pytest.approx(1.)
    assert r.raw_statistics['experimental_lp_route']['candidate'] == 'production'


def test_one_remaining_budget_and_no_retry(monkeypatch):
    import solverpilot
    from solverpilot.experimental import robust_lp
    p, m = case()
    seen = []
    def failing(*args, **kwargs):
        seen.append(kwargs['budget'].wall_time_s)
        raise RuntimeError('solver failed')
    ticks = iter([10., 10.1, 10.2, 10.3])
    monkeypatch.setattr(robust_lp, 'perf_counter', lambda: next(ticks))
    monkeypatch.setattr(solverpilot, 'solve_production', failing)
    with pytest.raises(RuntimeError, match='solver failed'):
        solve_robust_lp(p, m, environment_id='different', backends={}, time_limit_s=1.)
    assert len(seen) == 1 and seen[0] == pytest.approx(.7)


def test_setup_timeout_never_calls_solver(monkeypatch):
    from solverpilot.experimental import robust_lp
    p, m = case()
    ticks = iter([0., 1., 2., 3.])
    monkeypatch.setattr(robust_lp, 'perf_counter', lambda: next(ticks))
    with pytest.raises(TimeoutError):
        solve_robust_lp(p, m, environment_id='different', backends={}, time_limit_s=1.)
    for budget in (0, -1, True, np.inf, np.nan):
        with pytest.raises(ValueError):
            solve_robust_lp(p, m, environment_id='env', backends={}, time_limit_s=budget)
