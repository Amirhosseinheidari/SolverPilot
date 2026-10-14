from dataclasses import replace
import math
import pytest

from solverpilot.experimental.lp_session import LPRoutingSession
from solverpilot.experimental.lp_gain import calibrate_gain_guard
from solverpilot.experimental.robust_lp import decide_robust_lp
from test_lp_gain import calibration


def prepared():
    p, model, rows = calibration()
    guard = calibrate_gain_guard(model, rows, overhead_limit_s=.1)
    session = LPRoutingSession.create(model, guard, environment_id='env', cutoff_s=2.)
    return p, model, guard, session


def test_warm_session_never_reads_source_content(monkeypatch):
    p, model, guard, session = prepared()
    from pathlib import Path
    def forbidden(*a, **kw): raise AssertionError('repeated content read')
    monkeypatch.setattr(Path, 'read_bytes', forbidden)
    decision = decide_robust_lp(p, model, environment_id='env', available=('a',),
        gain_guard=guard, cutoff_s=2., session=session)
    assert decision.candidate == 'a'
    assert session.preparation_s >= 0


@pytest.mark.parametrize('changed', ['model', 'guard', 'environment', 'cutoff', 'source', 'missing'])
def test_binding_invalidated_on_change(monkeypatch, changed):
    from solverpilot.experimental import lp_session
    p, model, guard, session = prepared()
    environment, cutoff = 'env', 2.
    if changed == 'model': model = replace(model)
    if changed == 'guard': guard = replace(guard)
    if changed == 'environment': environment = 'changed'
    if changed == 'cutoff': cutoff = 1.
    if changed == 'source': monkeypatch.setattr(lp_session, '_snapshot', lambda p: ())
    if changed == 'missing':
        def missing(*a): raise FileNotFoundError()
        monkeypatch.setattr(lp_session, '_snapshot', missing)
    assert decide_robust_lp(p, model, environment_id=environment, available=('a',),
        gain_guard=guard, cutoff_s=cutoff, session=session).candidate == 'production'


@pytest.mark.parametrize('value', [-1., math.nan, math.inf, True])
def test_invalid_setup_rejected(value):
    p, model, guard, session = prepared()
    with pytest.raises(ValueError):
        decide_robust_lp(p, model, environment_id='env', available=('a',),
            gain_guard=guard, cutoff_s=2., session=session, setup_elapsed_s=value)


def test_stale_guard_cannot_create_session():
    p, model, guard, session = prepared()
    with pytest.raises(ValueError, match='implementation'):
        LPRoutingSession.create(model, replace(guard, implementation_sha256='0'*64),
                                environment_id='env', cutoff_s=2.)


def test_unavailable_candidate_falls_back():
    p, model, guard, session = prepared()
    assert decide_robust_lp(p, model, environment_id='env', available=(),
        gain_guard=guard, cutoff_s=2., session=session).candidate == 'production'


def test_backend_configuration_or_instance_change_invalidates_session():
    from solverpilot.backends import ScipyHighsLPBackend
    from solverpilot.experimental.robust_lp import solve_robust_lp
    p, model, guard, _ = prepared()
    backend = ScipyHighsLPBackend(method='highs-ds')
    backends = {'a': backend}
    session = LPRoutingSession.create(model, guard, environment_id='env', cutoff_s=2., backends=backends)
    assert session.backends_match(backends)
    assert not session.backends_match({'a': ScipyHighsLPBackend(method='highs-ds')})
    backend.method = 'highs-ipm'
    assert not session.backends_match(backends)
    result = solve_robust_lp(p, model, environment_id='env', backends=backends,
        gain_guard=guard, session=session)
    assert result.raw_statistics['experimental_lp_route']['candidate'] == 'production'


def test_version_two_leaf_guard_session_runs_actual_verified_solve():
    from solverpilot.backends import ScipyHighsLPBackend
    from solverpilot.experimental.robust_lp import solve_robust_lp
    from solverpilot.experimental.learned_lp_v2 import (
        lp_features_v2, fit_lp_selector_v2, calibrate_lp_guard_v2)
    from test_learned_lp_v2 import observations
    p, _, _ = calibration()
    features = lp_features_v2(p)
    train = [replace(r, features=features, samples={'a': ((.1, True),)*2, 'b': ((.2, True),)*2})
             for r in observations()]
    model = fit_lp_selector_v2(train, candidates=('a', 'b'), cutoff_s=3., protocol_sha256='a'*64)
    cal = [replace(r, features=features, samples={'a': ((.1, True),)*2,
           'b': ((.2, True),)*2, 'production': ((1., True),)*2}) for r in observations('calibration')]
    guard = calibrate_lp_guard_v2(model, cal, overhead_limit_s=.1)
    backends = {'a': ScipyHighsLPBackend(method='highs-ds'), 'b': ScipyHighsLPBackend(method='highs-ipm')}
    session = LPRoutingSession.create(model, guard, environment_id='env', cutoff_s=3., backends=backends)
    result = solve_robust_lp(p, model, environment_id='env', backends=backends,
        time_limit_s=3., gain_guard=guard, session=session)
    assert result.optimality_evidence.independently_verified_optimal
    assert result.raw_statistics['experimental_lp_route']['candidate'] == 'a'
    assert result.raw_statistics['experimental_lp_route']['leaf_id'] == 0
