from __future__ import annotations
import json, pickle
from pathlib import Path
import pytest
from solverpilot import inspect_problem
from solverpilot.experimental import decide_selective_lp_backend

ROOT=Path(__file__).resolve().parents[1]
MODEL=ROOT/'benchmarks/results/m27/m27-frozen-selector.json'
SPLIT=ROOT/'benchmarks/results/m26/m26-frozen-split.json'


def load_model(): return json.loads(MODEL.read_text())

def test_m27_model_is_frozen_before_test_and_development_gate_passed():
    m=load_model()
    assert m['test_outcomes_seen_during_fit'] is False
    assert m['development_gate_passed'] is True
    assert m['development_instances']==32 and m['test_instances']==16
    assert m['model_family']=='selective_cart_absolute_saving'
    assert m['selective_rule']['switch_margin_s']==0.005


def test_m27_split_test_remains_16_and_model_contains_no_test_names():
    s=json.loads(SPLIT.read_text()); names={r['instance'] for r in s['records'] if r['split']=='test'}
    assert len(names)==16
    raw=MODEL.read_text()
    assert all(n not in raw for n in names)


def test_m27_runtime_evaluator_falls_back_or_switches_without_sklearn_dependency():
    cache=Path('/tmp/m25_cache')
    if not cache.exists(): pytest.skip('M25 prepared cache unavailable')
    m=load_model()
    # Known development examples only; no held-out test outcomes are touched.
    p1=pickle.loads((cache/'seymour1.mps.gz.pkl').read_bytes())
    d1=decide_selective_lp_backend(inspect_problem(p1),m)
    assert d1.backend==m['default_backend']
    p2=pickle.loads((cache/'lotsize.mps.gz.pkl').read_bytes())
    d2=decide_selective_lp_backend(inspect_problem(p2),m)
    assert d2.backend in {m['default_backend'],m['alternate_backend']}


def test_m27_selector_rejects_non_lp():
    from solverpilot import LinearProblem, VariableDomain
    p=LinearProblem.from_data(A=[[1.0]],c=[1.0],variable_lower=[0.0],variable_upper=[1.0],constraint_lower=[0.0],constraint_upper=[1.0],domains=[VariableDomain.INTEGER])
    with pytest.raises(ValueError): decide_selective_lp_backend(inspect_problem(p),load_model())
