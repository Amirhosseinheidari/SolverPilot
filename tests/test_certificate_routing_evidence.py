"""Replay evidence integrity and claims without rerunning timing experiments."""
import hashlib
import json
from fractions import Fraction
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]/'docs/evidence/certificate-routing'


def read(name):
    return json.loads((ROOT/name).read_text(encoding='utf-8'))


def test_evidence_manifest_bytes_are_unchanged():
    manifest = read('manifest.json')
    for name, expected in manifest['files'].items():
        assert hashlib.sha256((ROOT/name).read_bytes()).hexdigest() == expected


def test_both_numerical_replays_and_exact_pilot_proof_keep_distinct_claims():
    for name in ('recovery-windows.json', 'recovery-wsl.json'):
        rows = read(name)['observations']
        assert {r['instance'] for r in rows} == {'netlib-pilot4', 'miplib-unitcal_7'}
        assert all(r['extended_check']['verified'] for r in rows)
        pilot = next(r for r in rows if r['instance'] == 'netlib-pilot4')
        assert not pilot['default_verified']
        assert 0 < pilot['extended_check']['gap'] < 1e-6
    exact = read('pilot4-exact.json')['result']
    assert exact['optimality_verified'] and exact['within_budget']
    assert exact['proof_kind'] == 'exact_optimal'
    assert Fraction(exact['exact_result']['absolute_gap']) == 0
    assert exact['exact_result']['solution_dimension'] == 1000


def test_development_outcomes_do_not_claim_promotion():
    summary = read('development-run/summary.json')
    assert summary['total_api_calls'] == 52 and summary['outcome_count'] == 20
    assert not summary['automatic_production_routing_enabled']
    assert not summary['production_promotion_supported']
    assert not read('development-run/guard.json')['certified_leaves']
    for scope in ('cold', 'amortized'):
        assert summary['cost_scopes'][scope]['v2']['verified_within_budget'] == 4
    measured = read('routing-session-installed.json')
    assert measured['setup_excluded_from_warm_decision'] is True
    assert measured['summary']['session']['switches'] == 8
    assert measured['eligible_summary']['session']['calls'] == 8


def test_balanced_development_replay_preserves_order_and_safe_fallback():
    prefix = 'development-balanced-run/'
    summary = read(prefix+'summary.json')
    assert summary['total_api_calls'] == 52
    assert not summary['automatic_production_routing_enabled']
    assert not summary['production_promotion_supported']
    assert not read(prefix+'guard.json')['certified_leaves']
    for split in ('train', 'calibration', 'test'):
        rows = read(prefix+split+'-outcomes.json')
        for name in {row['name'] for row in rows}:
            orders = [[row['strategy'] for row in rows
                       if row['name'] == name and row['repeat'] == repeat]
                      for repeat in (0, 1)]
            assert orders[1] == list(reversed(orders[0]))
    rows = read(prefix+'test-outcomes.json')
    assert len(rows) == 20 and all(row['verified'] for row in rows)
    assert sum(row['strategy'] == 'v2' for row in rows) == 4
