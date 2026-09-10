from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pytest

from solverpilot import (
    EvidenceClass, LinearProblem, PerformancePolicy,
    default_registry, plan_production_solve,
)
from solverpilot.benchmark.splits import validate_split_file

ROOT=Path(__file__).resolve().parents[1]
RES=ROOT/'benchmarks/results/m26'
from solverpilot.experimental import production_evidence_from_m26_validation


def load(name): return json.loads((RES/name).read_text())

def test_m26_split_is_exact_and_group_safe():
    cohort=load('../m25-cohort.json') if False else json.loads((ROOT/'benchmarks/results/m25-cohort.json').read_text())
    report=validate_split_file(RES/'m26-frozen-split-map.json', expected_instances=[x['instance'] for x in cohort['selected']])
    assert report.ok, report.errors
    assert report.instances==48
    assert report.split_counts=={'test':16,'train':20,'validation':12}

def test_m26_training_artifact_contains_no_test_costs():
    split=load('m26-frozen-split.json'); tv=load('m26-trainval-only.json')
    test_names={r['instance'] for r in split['records'] if r['split']=='test'}
    observed={r['instance'] for r in tv['rows']}
    assert tv['test_costs_present'] is False
    assert len(tv['rows'])==32
    assert test_names.isdisjoint(observed)
    raw=(RES/'m26-trainval-only.json').read_text()
    assert all(name not in raw for name in test_names)

def test_m26_model_is_frozen_without_test_outcomes_and_validation_failed():
    m=load('m26-frozen-model.json')
    assert m['model_family']=='ridge_log_cost_ratio'
    assert m['test_outcomes_seen_during_fit'] is False
    assert m['train_instances']==20 and m['validation_instances']==12
    assert m['validation_gate_passed'] is False
    assert m['validation']['policy_to_sbs'] > 1.0
    assert len(m['candidate_validation_summary'])==9
    assert all(c['policy_to_sbs']>1.0 for c in m['candidate_validation_summary'])

def test_m26_evidence_cannot_authorize_performance_ranking():
    model=load('m26-frozen-model.json')
    opportunity=json.loads((ROOT/'benchmarks/results/m25-public-ood-lp-opportunity-stability.json').read_text())
    ev=production_evidence_from_m26_validation(model, opportunity, source='M26 validation')
    assert ev.evidence_class is EvidenceClass.COMPARATIVE_VALIDATION
    assert ev.selection_opportunity_validated
    assert not ev.performance_ranking_validated
    assert not ev.heldout
    assert not ev.supports_performance_ranking

def test_m26_failed_validation_override_is_rejected():
    model=load('m26-frozen-model.json')
    opportunity=json.loads((ROOT/'benchmarks/results/m25-public-ood-lp-opportunity-stability.json').read_text())
    ev=production_evidence_from_m26_validation(model, opportunity, source='M26 validation')
    p=LinearProblem.from_data(A=[[1.0,1.0]],c=[1.0,2.0],variable_lower=[0.0,0.0],variable_upper=[1.0,1.0],constraint_lower=[1.0],constraint_upper=[np.inf])
    dec=plan_production_solve(p, default_registry(), evidence=ev, performance_policy=PerformancePolicy.ALLOW_COMPARATIVE, performance_override={'lp':'scipy-highs-ipm'})
    assert not dec.auto_performance_ranking_enabled
    assert dec.plan.selected_backend=='scipy-highs-ds'
    assert dec.rejected_performance_override_reason is not None

def test_m26_test_remains_sealed_no_test_result_artifact_exists():
    assert not (RES/'m26-test-results.json').exists()
    assert not (RES/'m26-test-evaluation.json').exists()
