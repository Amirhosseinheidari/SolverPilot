from __future__ import annotations

import json
from pathlib import Path

import pytest

from solverpilot import (
    EvidenceClass,
    LinearProblem,
    PerformancePolicy,
    ProductionEvidence,
    plan_production_solve,
)
from solverpilot.runtime import default_registry

from solverpilot.experimental import production_evidence_from_m24_public_ood


def _lp():
    return LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[float('inf')],
    )


def _negative_payload():
    return {
        'source':'MIPLIB 2017 benchmark-v2 derived continuous LP relaxations',
        'miplib_zip_sha256':'abc', 'instances':24,
        'pre_registered_rule':{
            'name':'ipm_if_m_over_n_ge_1.50',
            'origin':'M4/M5 synthetic LOFO/stability; frozen before M24 public OOD run',
        },
        'metrics':{
            'sbs_backend':'scipy-highs-ds',
            'objective_mismatches':0,
            'policy_to_sbs_bootstrap':{'point':1.018,'high':1.073},
        },
        'promotion_gates':{
            'minimum_instances':True,
            'objective_mismatches_zero':True,
            'policy_beats_sbs_point':False,
            'bootstrap_upper_below_1':False,
        },
        'selector_promoted':False,
        'environment':{'python':'x'},
        'claims_boundary':['public OOD','no superiority claim'],
    }


def test_failed_comparative_holdout_does_not_authorize_ranking():
    ev=production_evidence_from_m24_public_ood(_negative_payload(),source='unit')
    assert ev.evidence_class is EvidenceClass.COMPARATIVE_HELDOUT
    assert ev.heldout and ev.feature_cost_accounted and ev.fixed_environment
    assert ev.public_ood and ev.pre_registered_policy
    assert not ev.performance_ranking_validated
    assert not ev.supports_performance_ranking


def test_failed_m24_evidence_rejects_manual_performance_override():
    ev=production_evidence_from_m24_public_ood(_negative_payload(),source='unit')
    d=plan_production_solve(
        _lp(), default_registry(), evidence=ev,
        performance_policy=PerformancePolicy.ALLOW_COMPARATIVE,
        performance_override={'lp':'scipy-highs-ipm'},
    )
    assert d.plan.selected_backend == 'scipy-highs-ds'
    assert not d.auto_performance_ranking_enabled
    assert d.rejected_performance_override_reason is not None


def test_comparative_evidence_requires_public_preregistered_and_validated():
    base=dict(
        evidence_class=EvidenceClass.COMPARATIVE_HELDOUT, source='x',
        comparative_backends=('scipy-highs-ds','scipy-highs-ipm'),
        heldout=True, feature_cost_accounted=True, fixed_environment=True,
        public_ood=True, pre_registered_policy=True, selection_opportunity_validated=True, performance_ranking_validated=True,
    )
    assert ProductionEvidence(**base).supports_performance_ranking
    for field in ('public_ood','pre_registered_policy','selection_opportunity_validated','performance_ranking_validated'):
        row=dict(base); row[field]=False
        assert not ProductionEvidence(**row).supports_performance_ranking


def test_real_m24_result_is_negative_public_ood_evidence():
    p=Path(__file__).resolve().parents[1]/'benchmarks/results/m24-public-ood-lp-selector.json'
    if not p.exists(): pytest.skip('M24 benchmark result not present')
    payload=json.loads(p.read_text())
    ev=production_evidence_from_m24_public_ood(payload,source=str(p))
    assert payload['instances'] == 24
    assert payload['metrics']['objective_mismatches'] == 0
    assert payload['selector_promoted'] is False
    assert payload['metrics']['ds_wins'] >= 20
    assert not ev.supports_performance_ranking
