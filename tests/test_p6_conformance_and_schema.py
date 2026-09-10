from __future__ import annotations

import json
from importlib.resources import files

import pytest

import solverpilot as om


def test_conic_schema_is_packaged_and_matches_canonical_ir():
    schema=json.loads(files('solverpilot.conic').joinpath('conic-ir-v1.schema.json').read_text())
    assert schema['$id']=='solverpilot.conic-ir.v1'
    m=om.Model(); x=m.variable(1); t=m.variable(lower=0); m.soc(t,x); m.minimize(t)
    payload=m.compile(use_cache=False).execution_ir.to_canonical_dict()
    assert payload['schema']=='solverpilot.conic-ir.v1'
    assert payload['cones'][0]['kind']=='second_order'


def test_semantic_schema_bumps_only_for_conic_models():
    linear=om.Model(); x=linear.variable(); linear.minimize(x)
    assert linear._semantic_payload(include_parameter_values=False)['schema']=='solverpilot.semantic-model.p1.v1'
    conic=om.Model(); y=conic.variable(1); t=conic.variable(lower=0); conic.soc(t,y); conic.minimize(t)
    assert conic._semantic_payload(include_parameter_values=False)['schema']=='solverpilot.semantic-model.p6.v1'


def test_superscs_conformance_report_records_positive_and_negative_evidence():
    if not om.CasadiSuperSCSBackend().is_available():
        pytest.skip('CasADi SuperSCS plugin unavailable')
    report=om.conform_casadi_superscs_backend()
    assert report.passed
    names={c.name:c for c in report.checks}
    assert names['soc-3-4-5'].passed
    assert names['rotated-soc-analytic'].passed
    assert names['psd-fail-closed'].passed
    assert names['quadratic-conic-fail-closed'].passed


def test_psd_requirement_is_not_usable_for_superscs():
    m=om.Model(); t=m.variable(lower=0); m.psd(t*m.constant([[1.,0.],[0.,1.]])); m.minimize(t)
    p=m.compile(use_cache=False).execution_ir
    req=om.requirements_v2_for(p)
    manifest=om.CasadiSuperSCSBackend().capability_manifest_v2
    ok,checks=om.compatible_v2(manifest,req,allow_safe_emulation=True)
    assert not ok
    psd=[c for c in checks if c.key is om.CapabilityKey.CONSTRAINT_PSD][0]
    assert not psd.usable and psd.verification is om.VerificationLevel.UNVERIFIED
