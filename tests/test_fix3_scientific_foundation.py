from __future__ import annotations

from types import SimpleNamespace
import json
import numpy as np
import pytest

from solverpilot.benchmark import (
    CampaignSpec, SplitRecord, compute_scientific_metrics, fit_sbs_backend,
    fit_sbs_backend_from_split, fixed_backend_policy, generate_group_split,
    load_campaign_spec, miplib2017_campaign_template, qplib_campaign_template,
    save_campaign_spec, scientific_report_payload, seeded_random_policy,
    threshold_policy, validate_campaign, validate_split_records,
)
from solverpilot.benchmark.integrity import canonical_sha256, experiment_identity
from solverpilot.capabilities import CapabilityKey, ConvexityClass, IntegralityClass, ProblemClass, ProofRequirement, capability_requirements_for_descriptor, describe_problem, requirements_v2_for
from solverpilot.cp import CPLinearExprIR, CPProblem, ORToolsCPSATBackend
from solverpilot.evaluation import ProofLevel, assess_benchmark_row, assess_solve_result
from solverpilot.problem import LinearProblem, QuadraticProblem, VariableDomain


def _linear(integer=False):
    return LinearProblem.from_data(A=[[1.0,1.0]],c=[1.0,2.0],variable_lower=[0.0,0.0],variable_upper=[1.0,1.0],constraint_lower=[1.0],constraint_upper=[np.inf],domains=[VariableDomain.INTEGER if integer else VariableDomain.CONTINUOUS,VariableDomain.CONTINUOUS])

def _row(instance,backend,wall):
    return {"schema_version":"legacy-test","protocol_id":"p","environment_id":"e","run_id":f"{instance}-{backend}","instance":instance,"instance_sha256":canonical_sha256({"i":instance}),"backend":backend,"repetition":0,"state":"solved","validated":True,"reference_check":"objective_matches_optimum","public_status":"valid_optimal","objective":1.0,"wall_s":wall}


def test_campaign_spec_roundtrip_and_immutable_metadata(tmp_path):
    c=miplib2017_campaign_template(("scipy-highs-ds","scipy-highs-ipm"),cutoff_s=120.0); assert c.metadata["benchmark_instances"]==240 and len(c.sha256)==64
    with pytest.raises(TypeError): c.metadata["x"]=1
    path=tmp_path/'campaign.json'; save_campaign_spec(c,path); loaded=load_campaign_spec(path); assert loaded==c and loaded.sha256==c.sha256


def test_campaign_rejects_invalid_hash_and_timeout():
    with pytest.raises(ValueError): CampaignSpec("x","d","bad",("b",))
    with pytest.raises(ValueError): CampaignSpec("x","d","a"*64,("b",),cutoff_s=10,hard_timeout_s=9)


def test_group_split_is_deterministic_and_group_safe():
    mapping={"a1":"A","a2":"A","b1":"B","c1":"C","d1":"D","e1":"E","f1":"F"}; first=generate_group_split(mapping,seed=17); second=generate_group_split(dict(reversed(list(mapping.items()))),seed=17); assert first==second; assert validate_split_records(first,expected_instances=mapping).ok


def test_baselines_are_deterministic_and_explicit():
    a=seeded_random_policy(["z","a","m"],["b2","b1"],seed=5,overhead_s=.001); b=seeded_random_policy(["m","z","a"],["b1","b2"],seed=5,overhead_s=.001); assert a==b
    assert fixed_backend_policy(["a"],"highs")["a"]["overhead_s"]==0.0
    rule=threshold_policy({"a":{"n":2},"b":{"n":20}},feature="n",threshold=5,le_backend="ds",gt_backend="ipm",overhead_s=.002); assert rule["a"]["backend"]=="ds" and rule["b"]["backend"]=="ipm"


def test_sbs_fit_does_not_consume_test_instances():
    rows=[_row("tr","ds",1),_row("tr","ipm",2),_row("te","ds",10),_row("te","ipm",.1)]; fit=fit_sbs_backend(rows,training_instances=["tr"],cutoff_s=10); assert fit.selected_backend=="ds"
    records=(SplitRecord("tr","train","g1"),SplitRecord("te","test","g2")); assert fit_sbs_backend_from_split(rows,records,cutoff_s=10).selected_backend=="ds"
    with pytest.raises(ValueError,match="held-out test"): fit_sbs_backend_from_split(rows,records,fit_splits=("test",),cutoff_s=10)


def test_problem_ontology_lp_milp_qp_cp():
    lp=describe_problem(_linear(False)); assert lp.problem_class is ProblemClass.LP and lp.integrality is IntegralityClass.CONTINUOUS and lp.convexity is ConvexityClass.CONFIRMED_CONVEX
    milp=describe_problem(_linear(True)); assert milp.problem_class is ProblemClass.MILP and milp.proof_requirement is ProofRequirement.GLOBAL_OPTIMALITY
    qp=QuadraticProblem.from_data(P=np.eye(2),A=[[1,1]],q=[1,2],variable_lower=[0,0],variable_upper=[1,1],constraint_lower=[1],constraint_upper=[np.inf]); assert describe_problem(qp).problem_class is ProblemClass.CONVEX_QP
    cp=CPProblem(variables=(),intervals=(),constraints=(),objective=CPLinearExprIR()); assert describe_problem(cp).problem_class is ProblemClass.CP and CapabilityKey.PROBLEM_CP in requirements_v2_for(cp).required and CapabilityKey.PROBLEM_CP in capability_requirements_for_descriptor(describe_problem(cp)).required
    assert ORToolsCPSATBackend().capability_manifest_v2().claim(CapabilityKey.PROBLEM_CP).status.value=="supported"


def test_proof_levels_are_conservative():
    assert assess_benchmark_row({"public_status":"valid_optimal","validated":True}).level is ProofLevel.PRIMAL_VALIDATED
    e=SimpleNamespace(backend_reported_optimal=True,primal_validated=True,dual_verified=True,gap_verified=True,certificate_verified=False); a=assess_solve_result(SimpleNamespace(optimality_evidence=e)); assert a.level is ProofLevel.BOUND_VERIFIED and a.independently_verified_optimal


def test_scientific_report_enforces_oracle_boundary():
    c=miplib2017_campaign_template(("ds",)); summary={"instances":2,"backends":["ds"],"cost_field":"wall_s","sbs_solver":"ds","sbs_cost":1.0,"vbs":{"cost":.9,"is_oracle":True,"deployable":False},"sbs_vbs_gap":.1,"relative_sbs_vbs_gap":.1,"solver_stats":{},"auto_policy_metrics":None,"mixed_environments_allowed":False,"instance_identity_verified":True}
    p=scientific_report_payload(c,summary,campaign_run={"scientific_provenance_verified":True}); assert p["deployable_scientific_evidence"] is True
    bad=dict(summary); bad["vbs"]={"cost":.9,"is_oracle":True,"deployable":True}
    with pytest.raises(ValueError): scientific_report_payload(c,bad)


def test_qplib_template_is_explicitly_non_executable():
    q=qplib_campaign_template(("osqp",)); assert q.metadata["execution_supported"] is False and q.metadata["supported_subset_count"] is None


def test_scientific_metrics_par2_par10_no_invented_gap():
    rows=[_row("a","ds",1),_row("a","ipm",2),_row("b","ds",3),_row("b","ipm",1)]; m=compute_scientific_metrics(rows,cutoff_s=10); assert m["par10"]["vbs_cost"]==pytest.approx(1.0); assert m["mean_reported_relative_gap"]=={"ds":None,"ipm":None}
