from dataclasses import fields

import pytest

from solverpilot import LinearProblem, solve
from solverpilot.backends.base import BackendSolveResult
from solverpilot.backends.pdlp import PDLPBackend
from solverpilot.backends.cuopt import CuOptBackend
from solverpilot.exceptions import BackendUnavailableError
from solverpilot.runtime.auto import builtin_backend_candidates, explicit_optional_backend
from solverpilot.runtime.evidence import save_evidence_bundle, replay_evidence_bundle
from solverpilot.runtime.manifest import environment, run_manifest, _replay_payload


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[1.], constraint_upper=[float('inf')])


@pytest.mark.parametrize('cls,settings', [
    (PDLPBackend, dict(threads=3, absolute_tolerance=2e-8, relative_tolerance=3e-8, iteration_limit=321)),
    (CuOptBackend, dict(threads=3, tolerance=2e-8, iteration_limit=321)),
])
def test_optional_replay_restores_exact_configuration_and_new_identity(monkeypatch, tmp_path, cls, settings):
    if cls is PDLPBackend:
        import solverpilot.backends.pdlp as pdlp
        monkeypatch.setattr(pdlp, 'version', lambda _: 'fixture-version')
    calls = []
    monkeypatch.setattr(cls, 'is_available', lambda _: True)
    def execute(self, p):
        calls.append({f.name: getattr(self, f.name) for f in fields(self) if f.init})
        return BackendSolveResult('optimal', [1.], 1., {'canonical_dual': [-1., 0.]})
    monkeypatch.setattr(cls, 'solve', execute)
    p = problem()
    r = solve(p, backend=cls(**settings))
    path = tmp_path/'evidence.json'
    save_evidence_bundle(path, p, r, include_model=True)
    replay = replay_evidence_bundle(path)
    assert calls[0] == calls[1]
    assert replay.trace.replay_of == r.execution_id
    assert replay.execution_id != r.execution_id
    assert replay.optimality_evidence.independently_verified_optimal
    assert not any(isinstance(b, cls) for b in builtin_backend_candidates())


def test_actual_pdlp_evidence_can_replay(tmp_path):
    pytest.importorskip('ortools')
    p = problem()
    r = solve(p, backend='ortools-pdlp')
    path = tmp_path/'pdlp.json'
    save_evidence_bundle(path, p, r, include_model=True)
    replay = replay_evidence_bundle(path)
    assert replay.validation.valid
    assert replay.objective == pytest.approx(1., abs=1e-7)
    assert replay.trace.replay_of == r.execution_id


def test_unavailable_optional_replay_and_unknown_factory_fail_closed(monkeypatch):
    p = problem()
    r = solve(p, backend='scipy-highs-ds')
    payload = run_manifest(p, r, include_model=True)
    from solverpilot.runtime.manifest import _seal
    payload['backend'] = 'ortools-pdlp'
    payload['parameters']['backend_configuration'] = {}
    monkeypatch.setattr(PDLPBackend, 'is_available', lambda _: False)
    with pytest.raises(BackendUnavailableError):
        _replay_payload(_seal(payload))
    assert explicit_optional_backend('unregistered-arbitrary-code') is None


def test_cuda_binding_version_drift_blocks_replay(monkeypatch):
    import solverpilot.runtime.manifest as manifest
    actual_version = manifest.version
    binding = ['26.8.0']
    def version(name):
        return binding[0] if name == 'cuopt-cu12' else actual_version(name)
    monkeypatch.setattr(manifest, 'version', version)
    environment.cache_clear()
    try:
        p = problem()
        r = solve(p, backend='scipy-highs-ds')
        payload = run_manifest(p, r, include_model=True)
        assert payload['environment']['packages']['cuopt-cu12'] == '26.8.0'
        binding[0] = '26.10.0'
        environment.cache_clear()
        with pytest.raises(ValueError, match='package versions differ'):
            _replay_payload(payload)
    finally:
        environment.cache_clear()
