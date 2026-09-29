"""Integrity/accounting replay of captured public results, without native solves."""
import hashlib
import importlib.util
import json
from pathlib import Path

import pytest

ROOT=Path(__file__).resolve().parents[1]
EVIDENCE=ROOT/"docs/evidence/public-lp-local"


def read(name):
    return json.loads((EVIDENCE/name).read_text())


def test_frozen_files_and_model_binding():
    for name,digest in read('manifest.json')['files'].items():
        assert hashlib.sha256((EVIDENCE/name).read_bytes()).hexdigest()==digest
    protocol=read('protocol.json')
    assert protocol['model_payload_sha256']==json.loads((ROOT/'docs/evidence/learned-lp-local/model.json').read_text())['sha256']
    assert protocol['cohort_sha256']==hashlib.sha256((EVIDENCE/'cohort.json').read_bytes()).hexdigest()


def test_admission_excludes_known_instances_and_groups():
    cohort=read('cohort.json'); selected=cohort['selected']; prior=set(cohort['prior_audit']['names'])
    assert len(selected)==len({c['data_hash'] for c in selected})==24
    assert len({c['group'] for c in selected})==24
    assert not {c['name'].lower() for c in selected}&prior
    assert all(c['reader_agreement'] for c in selected)
    assert all(c['reference'] is None for c in selected if c['source']=='miplib')
    for source in ('netlib','miplib'):
        assert sum(c['source']==source for c in selected)==12


def assert_equal(left,right):
    if isinstance(left,dict):
        assert left.keys()==right.keys()
        for key in left:assert_equal(left[key],right[key])
    elif isinstance(left,list):
        assert len(left)==len(right)
        for a,b in zip(left,right):assert_equal(a,b)
    elif type(left) is float:
        assert left==pytest.approx(right,rel=1e-12,abs=1e-12)
    else:assert left==right


def test_full_public_gate_replays_from_all_repetitions():
    spec=importlib.util.spec_from_file_location('qualify_public_lp',ROOT/'benchmarks/qualify_public_lp.py')
    module=importlib.util.module_from_spec(spec);spec.loader.exec_module(module)
    rows=read('observations.json');cases=read('cohort.json')['selected']
    assert len(rows)==288
    assert_equal(module.summarize(rows,cases),read('summary.json'))
    for row in rows:
        if row['strategy']=='learned' and 'decision' in row:
            assert row['model_sha256']==read('protocol.json')['model_payload_sha256']


def test_stress_cases_do_not_promote_terminal_claims_to_optimality():
    rows=read('stress.json')
    assert len(rows)==24
    assert {r['case'] for r in rows}=={'infeasible','unbounded','ill_scaled','deadline'}
    for row in rows:
        if row['case'] in {'infeasible','unbounded'}:
            assert row['verified'] is False
    assert read('summary.json')['automatic_production_routing_enabled'] is False
