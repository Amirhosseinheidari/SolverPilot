from dataclasses import replace

import pytest

from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.experimental.lp_session import LPRoutingSession
from solverpilot.experimental.lp_gain import calibrate_gain_guard
from solverpilot.experimental.robust_lp import solve_robust_lp
from test_lp_gain import calibration


def setup():
    p, model, rows = calibration()
    guard = calibrate_gain_guard(model, rows, overhead_limit_s=.1)
    backends = {'a': ScipyHighsLPBackend(method='highs-ds')}
    session = LPRoutingSession.create(model, guard, environment_id='env', cutoff_s=2., backends=backends)
    return p, model, guard, backends, session


def test_warm_calls_use_cached_artifact_hashes_and_consistent_total_timing(monkeypatch):
    p, model, guard, backends, session = setup()
    identities = session.artifact_digests(model, guard)
    monkeypatch.setattr(type(model), 'payload', lambda _: pytest.fail('warm model serialization'))
    monkeypatch.setattr(type(guard), 'payload', lambda _: pytest.fail('warm guard serialization'))
    r = solve_robust_lp(p, model, environment_id='env', backends=backends, gain_guard=guard, session=session)
    route = r.raw_statistics['experimental_lp_route']
    assert route['candidate'] == 'a'
    assert (route['model_sha256'], route['gain_guard_sha256']) == identities
    assert route['call_wall_s'] == r.raw_statistics['call_budget']['elapsed_s'] == r.trace.timings.total_s


def test_replacement_artifacts_never_borrow_cached_identities():
    _, model, guard, _, session = setup()
    assert session.artifact_digests(replace(model), guard) is None
    assert session.artifact_digests(model, replace(guard)) is None


def test_mutable_nested_identity_cannot_create_a_cached_session():
    _, model, guard, backends, _ = setup()
    mutable = replace(model, training_instances=(['mutable'],))
    with pytest.raises(ValueError, match='immutable'):
        LPRoutingSession.create(mutable, guard, environment_id='env', cutoff_s=2., backends=backends)


def test_implementation_binding_includes_verifiers_and_isolation_helpers():
    from solverpilot.experimental.lp_gain import implementation_paths
    paths = dict(implementation_paths())
    assert 'validate/core.py' in paths
    assert 'runtime/result.py' in paths
    assert 'backends/_highs_thread_worker.py' in paths


def test_stateless_digest_work_consumes_overhead_allowance(monkeypatch):
    from solverpilot.experimental import robust_lp
    p, model, guard, backends, _ = setup()
    clock = [10.]
    actual = type(guard).payload
    def expensive_payload(self):
        clock[0] += .2
        return actual(self)
    monkeypatch.setattr(type(guard), 'payload', expensive_payload)
    monkeypatch.setattr(robust_lp, 'perf_counter', lambda: clock[0])
    r = solve_robust_lp(p, model, environment_id='env', backends=backends, gain_guard=guard)
    route = r.raw_statistics['experimental_lp_route']
    assert route['candidate'] == 'production' and route['reason'] == 'gain_guard_fallback'
    assert route['overhead_s'] == pytest.approx(.2)
