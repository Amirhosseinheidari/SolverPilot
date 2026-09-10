from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import numpy as np

from solverpilot import LinearProblem
from solverpilot.experimental import production_evidence_from_m29_value_audit

ROOT=Path(__file__).resolve().parents[1]
AUDIT=ROOT/'benchmarks/results/m29/m29-value-of-information-audit.json'
FEATURES=ROOT/'benchmarks/results/m29/m29-feature-extraction.json'
PROTOCOL=ROOT/'docs/research/M29-VALUE-OF-INFORMATION-PROTOCOL.md'
WORKER=ROOT/'benchmarks/m29_feature_worker.py'
TARGET_STABILITY=ROOT/'benchmarks/results/m29/m29-target-stability-audit.json'
SCALING_PROPERTY=ROOT/'benchmarks/results/m29/m29-scaling-invariance-property.json'


def _load_worker_module():
    spec=importlib.util.spec_from_file_location('m29_feature_worker_test',WORKER)
    assert spec and spec.loader
    mod=importlib.util.module_from_spec(spec); spec.loader.exec_module(mod); return mod


def test_m29_is_discovery_not_promotion_and_closes_m30_gate():
    p=json.loads(AUDIT.read_text())
    assert p['schema']=='optimind.m29.value_of_information_audit.v1'
    assert p['cohorts']=={'m25':48,'m28':48,'overlap':0}
    assert p['m30_authorized'] is False
    assert p['m30_authorized_families']==[]
    assert p['production_performance_routing_authorized'] is False
    assert p['representation_audit_integrity_passed'] is True
    assert p['selection_opportunity_context_validated'] is True
    assert all(p['source_integrity'].values())
    assert all(not row['passes_all'] for row in p['family_authorization'].values())
    ev=production_evidence_from_m29_value_audit(p,source='M29 raw')
    assert ev.selection_opportunity_validated
    assert not ev.heldout
    assert not ev.performance_ranking_validated
    assert not ev.supports_performance_ranking


def test_m29_feature_extraction_is_complete_and_probe_is_not_solver_trajectory():
    p=json.loads(FEATURES.read_text())
    assert p['counts']['m25']=={'total':48,'ok':48,'probe_ok':48}
    assert p['counts']['m28']=={'total':48,'ok':48,'probe_ok':48}
    for row in p['rows']:
        assert row['ok']
        assert not row['probe_error']
        assert set(row['features']['D'])=={
            'presolved_n','presolved_m','presolved_nnz',
            'presolve_col_reduction_fraction','presolve_row_reduction_fraction','presolve_nnz_reduction_fraction',
        }
        assert not any('iteration' in k or 'simplex' in k or 'ipm_' in k for k in row['features']['D'])


def test_m29_canonical_static_features_are_invariant_to_positive_row_and_objective_scaling():
    mod=_load_worker_module()
    A=np.array([[1.0,-2.0,0.0],[0.0,3.0,4.0]])
    p=LinearProblem.from_data(
        A=A,c=[2.0,-5.0,0.0],
        variable_lower=[0.0,-np.inf,1.0],variable_upper=[10.0,np.inf,7.0],
        constraint_lower=[-np.inf,2.0],constraint_upper=[8.0,2.0],
    )
    scales=np.array([10.0,0.1])
    q=LinearProblem.from_data(
        A=A*scales[:,None],c=np.array([2.0,-5.0,0.0])*100.0,
        variable_lower=p.variable_lower,variable_upper=p.variable_upper,
        constraint_lower=np.where(np.isfinite(p.constraint_lower),p.constraint_lower*scales,p.constraint_lower),
        constraint_upper=np.where(np.isfinite(p.constraint_upper),p.constraint_upper*scales,p.constraint_upper),
    )
    a=mod.static_features(p); b=mod.static_features(q)
    assert a.keys()==b.keys()
    for k in a:
        assert np.isclose(a[k],b[k],rtol=1e-10,atol=1e-12), (k,a[k],b[k])


def test_m29_protocol_explicitly_forbids_using_audit_as_promotion_evidence():
    text=PROTOCOL.read_text().lower()
    assert 'post-outcome representation audit' in text
    assert 'fresh held-out promotion evidence' in text
    assert 'can never enable production performance routing by itself' in text


def test_m29_target_stability_shows_noise_is_not_complete_explanation():
    p=json.loads(TARGET_STABILITY.read_text())
    assert p['schema']=='optimind.m29.target_stability.v1'
    assert p['m25']['delta_rank_spearman_run1_vs_run2'] >= 0.90
    assert p['m25']['winner_sign_agreement_fraction'] >= 0.90
    assert p['m28']['median_round_pair_delta_spearman'] >= 0.90
    assert p['interpretation']['target_noise_is_not_a_complete_explanation']
    assert p['interpretation']['opportunity_not_only_single_outlier_m25']
    assert p['interpretation']['opportunity_not_only_single_outlier_m28']


def test_m29_scaling_invariance_property_campaign_passed():
    p=json.loads(SCALING_PROPERTY.read_text())
    assert p['schema']=='optimind.m29.scaling_invariance_property.v1'
    assert p['cases']==200
    assert p['passed']==200
    assert p['failed']==0


def test_m29_evidence_never_becomes_performance_authority_even_if_audit_is_tampered():
    p=json.loads(AUDIT.read_text())
    p['m30_authorized']=True
    p['m30_authorized_families']=['ABCD']
    p['production_performance_routing_authorized']=True
    ev=production_evidence_from_m29_value_audit(p,source='tampered M29')
    assert ev.selection_opportunity_validated
    assert not ev.heldout
    assert not ev.performance_ranking_validated
    assert not ev.supports_performance_ranking


def test_m29_missing_opportunity_context_fails_closed():
    p=json.loads(AUDIT.read_text())
    p.pop('selection_opportunity_context_validated',None)
    ev=production_evidence_from_m29_value_audit(p,source='missing-context M29')
    assert not ev.selection_opportunity_validated
    assert not ev.supports_performance_ranking
