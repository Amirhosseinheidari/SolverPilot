"""Replay frozen training and held-out accounting without rerunning solvers."""
import hashlib
import importlib
import json
from pathlib import Path

import pytest

from solverpilot.experimental.learned_lp import LPSelector, fit_lp_selector

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT/'docs/evidence/robust-lp-local'


def read(name):
    return json.loads((EVIDENCE/name).read_text())


@pytest.fixture
def runner(monkeypatch):
    monkeypatch.syspath_prepend(str(ROOT/'benchmarks'))
    return importlib.import_module('qualify_robust_lp')


def test_manifest_model_and_cohort_bindings():
    for name, digest in read('manifest.json')['files'].items():
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest() == digest
    protocol = read('protocol.json')
    model = LPSelector.load(EVIDENCE/'model.json')
    assert model.protocol_sha256 == hashlib.sha256((EVIDENCE/'protocol.json').read_bytes()).hexdigest()
    assert read('model-freeze.json')['sha256'] == hashlib.sha256((EVIDENCE/'model.json').read_bytes()).hexdigest()
    for prefix in ('training', 'heldout'):
        assert protocol[prefix+'_cohort_sha256'] == hashlib.sha256((EVIDENCE/(prefix+'-cohort.json')).read_bytes()).hexdigest()


def test_public_training_replays_without_heldout_outcomes(runner):
    protocol = read('protocol.json')
    rows = runner.training_rows(read('training-observations.json'), read('training-cohort.json')['selected'],
                                protocol['environment_id'])
    model = fit_lp_selector(rows, candidates=runner.CANDIDATES, cutoff_s=12.,
        protocol_sha256=hashlib.sha256((EVIDENCE/'protocol.json').read_bytes()).hexdigest(),
        min_leaf_groups=4, switch_margin_s=.005)
    assert model == LPSelector.load(EVIDENCE/'model.json')


def assert_close(left, right):
    if isinstance(left, dict):
        assert left.keys() == right.keys()
        for k in left: assert_close(left[k], right[k])
    elif isinstance(left, list):
        assert len(left) == len(right)
        for a, b in zip(left, right): assert_close(a, b)
    elif type(left) is float:
        assert left == pytest.approx(right, rel=1e-12, abs=1e-12)
    else:
        assert left == right


def test_fresh_split_and_complete_heldout_gate_replay(runner):
    training, heldout = read('training-cohort.json')['selected'], read('heldout-cohort.json')['selected']
    runner.verify_split(training, heldout)
    prior = set(read('heldout-cohort.json')['prior_audit']['names'])
    assert not {c['name'].lower() for c in heldout} & prior
    rows = read('heldout-observations.json')
    assert len(rows) == 288 and len(read('training-observations.json')) == 144
    assert_close(runner.public_summary(rows, heldout), read('summary.json'))
    assert read('summary.json')['automatic_production_routing_enabled'] is False
    for r in rows:
        if r['strategy'] == 'robust' and 'route' in r:
            assert r['route']['model_sha256'] == read('model.json')['sha256']
            if r['route']['reason'] == 'production_fallback':
                assert r['route']['candidate'] == 'production'


def test_stress_does_not_manufacture_optimality_for_terminal_claims():
    rows = read('stress.json')
    assert len(rows) == 24
    for r in rows:
        if r['case'] in {'infeasible', 'unbounded', 'deadline'}:
            assert r['verified'] is False
