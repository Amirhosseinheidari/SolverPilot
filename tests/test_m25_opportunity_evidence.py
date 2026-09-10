from __future__ import annotations
import json
from pathlib import Path

from solverpilot import (
    EvidenceClass, LinearProblem, PerformancePolicy,
    plan_production_solve,
)
from solverpilot.runtime import default_registry

from solverpilot.experimental import production_evidence_from_m25_opportunity


def _lp():
    return LinearProblem.from_data(
        A=[[1.0,1.0]], c=[1.0,2.0],
        variable_lower=[0.0,0.0], variable_upper=[1.0,1.0],
        constraint_lower=[1.0], constraint_upper=[float('inf')],
    )


def _positive():
    metric={'objective_mismatches':0,'oracle_with_inspection_to_sbs_bootstrap':{'high':0.95}}
    return {
        'same_cohort':True,'same_cohort_signature':True,'opportunity_validated_both':True,
        'run1':{'metrics':metric,'validated':True},'run2':{'metrics':metric,'validated':True},
        'environment_audit':{'captured_in_same_active_container_session':True},
    }


def test_positive_opportunity_is_not_selector_authority():
    ev=production_evidence_from_m25_opportunity(_positive(),source='unit')
    assert ev.evidence_class is EvidenceClass.COMPARATIVE_HELDOUT
    assert ev.selection_opportunity_validated
    assert not ev.performance_ranking_validated
    assert not ev.pre_registered_policy
    assert not ev.supports_performance_ranking


def test_opportunity_evidence_cannot_override_lp_baseline():
    ev=production_evidence_from_m25_opportunity(_positive(),source='unit')
    d=plan_production_solve(_lp(),default_registry(),evidence=ev,performance_policy=PerformancePolicy.ALLOW_COMPARATIVE,performance_override={'lp':'scipy-highs-ipm'})
    assert d.plan.selected_backend == 'scipy-highs-ds'
    assert not d.auto_performance_ranking_enabled
    assert d.rejected_performance_override_reason is not None


def test_negative_or_unstable_opportunity_is_not_validated():
    p=_positive(); p['same_cohort_signature']=False
    ev=production_evidence_from_m25_opportunity(p,source='unit')
    assert not ev.selection_opportunity_validated
    assert not ev.supports_performance_ranking


def test_real_m25_stability_is_positive_opportunity_only():
    p=Path(__file__).resolve().parents[1]/'benchmarks/results/m25-public-ood-lp-opportunity-stability.json'
    payload=json.loads(p.read_text())
    ev=production_evidence_from_m25_opportunity(payload,source=str(p))
    assert payload['opportunity_validated_both'] is True
    assert ev.selection_opportunity_validated
    assert not ev.supports_performance_ranking
    assert payload['run1']['metrics']['oracle_with_inspection_relative_gain'] > 0.09
    assert payload['run2']['metrics']['oracle_with_inspection_relative_gain'] > 0.09
