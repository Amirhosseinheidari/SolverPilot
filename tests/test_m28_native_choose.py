from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from solverpilot.experimental import production_evidence_from_m28_native_choose
from solverpilot.backends import ScipyHighsLPBackend

ROOT = Path(__file__).resolve().parents[1]
RESULT = ROOT / 'benchmarks/results/m28/m28-native-choose-audit.json'
COHORT = ROOT / 'benchmarks/results/m28/m28-fresh-cohort.json'
SEMANTICS = ROOT / 'benchmarks/results/m28/m28-scipy-highs-semantics.json'
M25 = ROOT / 'benchmarks/results/m25-cohort.json'
PROTOCOL = ROOT / 'docs/research/M28-NATIVE-CHOOSE-BASELINE-PROTOCOL.md'
RUNNER = ROOT / 'docs/history/frozen-runners/m28/m28_run_shard.py'
WORKER = ROOT / 'docs/history/frozen-runners/m28/m28_native_choose_worker.py'


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_scipy_highs_lp_backend_does_not_expose_ambiguous_highs_auto_mode():
    with pytest.raises(ValueError):
        ScipyHighsLPBackend(method='highs')


def test_m28_scipy_semantics_probe_fails_closed_on_local_117_behavior():
    p = json.loads(SEMANTICS.read_text())
    assert p['scipy_version'] == '1.17.0'
    assert p['public_highs_maps_to_solver_none'] is True
    assert p['private_highs_none_documented_as_simplex'] is True
    assert p['authority_for_performance_ranking'] is False


def test_m28_fresh_cohort_has_zero_overlap_with_m25_selected():
    c = json.loads(COHORT.read_text())
    old = json.loads(M25.read_text())
    fresh = {x['instance'] for x in c['selected']}
    prior = {x['instance'] for x in old['selected']}
    assert len(fresh) == 48
    assert len(prior) == 48
    assert fresh.isdisjoint(prior)
    assert c['fresh_overlap_count'] == 0


def test_m28_campaign_accounting_and_provenance_hashes_are_complete():
    p = json.loads(RESULT.read_text())
    assert p['instances'] == 48
    assert p['rounds'] == 3
    assert p['pre_registered_gates']['complete_three_round_accounting'] is True
    assert p['pre_registered_gates']['single_fixed_host_and_runner_provenance'] is True
    hashes = p['file_hashes']
    assert hashes['protocol_sha256'] == sha(PROTOCOL)
    assert hashes['runner_sha256'] == sha(RUNNER)
    assert hashes['worker_sha256'] == sha(WORKER)
    for solver in ('choose', 'simplex', 'ipm'):
        count = sum(len(r['samples'][solver]) for r in p['rows'])
        assert count == 48 * 3


def test_m28_native_choose_is_not_misrepresented_as_cross_instance_selector():
    p = json.loads(RESULT.read_text())
    m = p['metrics']
    assert m['both_simplex_and_ipm_observed_under_choose'] is False
    assert m['choose_algorithm_diagnostics'].get('ipm', 0) == 0
    assert m['choose_algorithm_diagnostics'].get('simplex', 0) >= 1
    assert p['native_choose_closes_portfolio_opportunity'] is False
    assert m['forced_sbs_to_vbs_relative_gap'] > 0.03
    assert m['choose_vbs_gap_closure'] < 0.80


def test_m28_invalid_candidates_fail_closed_instead_of_becoming_terminal_success():
    p = json.loads(RESULT.read_text())
    assert p['pre_registered_gates']['zero_independent_invalidity'] is False
    assert p['metrics']['invalid_counts']['choose'] > 0
    assert p['metrics']['invalid_counts']['simplex'] > 0
    assert p['metrics']['invalid_counts']['ipm'] > 0


def test_m28_evidence_can_validate_opportunity_without_authorizing_ranking():
    p = json.loads(RESULT.read_text())
    ev = production_evidence_from_m28_native_choose(p, source='M28 raw')
    assert ev.public_ood is True
    assert ev.fixed_environment is True
    assert ev.selection_opportunity_validated is True
    assert ev.performance_ranking_validated is False
    assert ev.supports_performance_ranking is False
    # Campaign independent-validation gate is intentionally false because some
    # time-limit/near-tolerance candidates were rejected by the canonical validator.
    assert ev.independent_validation_passed is False
