from __future__ import annotations
import json
from pathlib import Path
from solverpilot import (
    EvidenceClass, LinearProblem, PerformancePolicy, default_registry,
    plan_production_solve,
)

ROOT=Path(__file__).resolve().parents[1]
M27=ROOT/'benchmarks/results/m27'
M26=ROOT/'benchmarks/results/m26'
from solverpilot.experimental import production_evidence_from_m27_heldout


def load(p): return json.loads(p.read_text())

def evidence():
    return production_evidence_from_m27_heldout(
        load(M27/'m27-frozen-selector.json'),
        load(M27/'m27-one-time-heldout-test.json'),
        load(ROOT/'benchmarks/results/m25-public-ood-lp-opportunity-stability.json'),
        source='M27 real held-out test',
    )

def test_m27_real_test_is_consumed_complete_and_negative():
    p=load(M27/'m27-one-time-heldout-test.json')
    assert p['test_was_sealed_until_this_execution'] is True
    assert p['test_execution_is_one_time'] is True
    assert p['metrics']['outcomes']==64
    assert p['metrics']['worker_errors']==0
    assert p['metrics']['objective_mismatches']==0
    assert p['performance_ranking_validated'] is False
    assert p['metrics']['policy_to_sbs'] > 1.0
    assert p['metrics']['bootstrap_policy_to_sbs']['high'] > 1.0


def test_m27_negative_heldout_never_authorizes_ranking():
    ev=evidence()
    assert ev.evidence_class is EvidenceClass.COMPARATIVE_HELDOUT
    assert ev.heldout and ev.feature_cost_accounted and ev.public_ood and ev.pre_registered_policy
    assert ev.selection_opportunity_validated
    assert not ev.performance_ranking_validated
    assert not ev.supports_performance_ranking
    # The consumed M27 payload did not capture an explicit fixed-host fingerprint.
    # Negative evidence stays valid for falsification, but must not overclaim
    # deployment-grade timing provenance.
    assert ev.fixed_environment is False
    assert ev.corpus_integrity_passed is False


def test_m27_negative_evidence_rejects_speed_override():
    ev=evidence()
    p=LinearProblem.from_data(A=[[1.0,1.0]],c=[1.0,2.0],variable_lower=[0.0,0.0],variable_upper=[1.0,1.0],constraint_lower=[1.0],constraint_upper=[float('inf')])
    d=plan_production_solve(p,default_registry(),evidence=ev,performance_policy=PerformancePolicy.ALLOW_COMPARATIVE,performance_override={'lp':'scipy-highs-ipm'})
    assert d.plan.selected_backend=='scipy-highs-ds'
    assert not d.auto_performance_ranking_enabled
    assert d.rejected_performance_override_reason is not None


def test_m27_model_protocol_hashes_are_frozen_and_match_test():
    m=load(M27/'m27-frozen-selector.json'); t=load(M27/'m27-one-time-heldout-test.json')
    assert t['model_sha256']==m['model_sha256']
    assert t['protocol_sha256']==m['protocol_sha256']
    assert m['test_outcomes_seen_during_fit'] is False


def test_m27_tampered_model_cannot_claim_heldout_integrity():
    model=load(M27/'m27-frozen-selector.json')
    model['leaf_stats']['2']['mean_saving_s'] += 1.0
    ev=production_evidence_from_m27_heldout(
        model,
        load(M27/'m27-one-time-heldout-test.json'),
        load(ROOT/'benchmarks/results/m25-public-ood-lp-opportunity-stability.json'),
        source='tamper-test',
    )
    assert ev.heldout is False
    assert not ev.supports_performance_ranking
