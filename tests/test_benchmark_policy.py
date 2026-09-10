from __future__ import annotations

import json

import pytest

from solverpilot.benchmark.policy import evaluate_policy_map, load_policy_map


def _rows():
    rows=[]
    vals={"i1":{"a":1.0,"b":2.0},"i2":{"a":4.0,"b":1.0}}
    k=0
    for inst,bs in vals.items():
        for backend,cost in bs.items():
            k+=1
            rows.append({"run_id":str(k),"protocol_id":"p","environment_id":"e","instance_sha256":("a" if inst=="i1" else "b")*64,"instance":inst,"backend":backend,"repetition":0,"state":"solved","validated":True,"reference_check":"not_checkable","public_status":"valid_optimal","optimality_evidence":{"independently_verified_optimal":True},"wall_s":cost})
    return rows


def test_external_policy_can_match_vbs():
    policy={"i1":{"backend":"a","overhead_s":0},"i2":{"backend":"b","overhead_s":0}}
    out=evaluate_policy_map(_rows(),policy,cutoff_s=10,bootstrap_draws=100)
    assert out["policy_metrics"]["gap_closure"]==1.0


def test_policy_overhead_is_counted():
    base={"i1":{"backend":"a","overhead_s":0},"i2":{"backend":"b","overhead_s":0}}
    costly={k:{**v,"overhead_s":1.0} for k,v in base.items()}
    a=evaluate_policy_map(_rows(),base,cutoff_s=10,bootstrap_draws=50)
    b=evaluate_policy_map(_rows(),costly,cutoff_s=10,bootstrap_draws=50)
    assert b["policy_metrics"]["policy_cost"] > a["policy_metrics"]["policy_cost"]


def test_policy_requires_exact_instance_coverage():
    with pytest.raises(ValueError,match="exactly cover"):
        evaluate_policy_map(_rows(),{"i1":{"backend":"a","overhead_s":0}},cutoff_s=10)


def test_policy_loader_accepts_short_form_but_marks_overhead_unmeasured(tmp_path):
    p=tmp_path/"p.json"; p.write_text(json.dumps({"i1":"a"}))
    assert load_policy_map(p)=={"i1":{"backend":"a","overhead_s":None,"overhead_measured":False,"overhead_components_s":{}}}

def test_policy_rejects_unmeasured_overhead_by_default(tmp_path):
    p=tmp_path/"p.json"; p.write_text(json.dumps({"i1":"a","i2":"b"}))
    policy=load_policy_map(p)
    with pytest.raises(ValueError,match="overhead is unmeasured"):
        evaluate_policy_map(_rows(),policy,cutoff_s=10,bootstrap_draws=50)
    out=evaluate_policy_map(_rows(),policy,cutoff_s=10,bootstrap_draws=50,allow_unmeasured_overhead=True)
    assert out["overhead_accounting"]["complete"] is False
    assert out["cost_accounting"]["deployable_evidence"] is False

def test_policy_component_breakdown_must_match_total(tmp_path):
    p=tmp_path/"p.json"; p.write_text(json.dumps({"i1":{"backend":"a","overhead_s":1.0,"overhead_components_s":{"feature":0.2}}}))
    with pytest.raises(ValueError,match="disagrees"):
        load_policy_map(p)
