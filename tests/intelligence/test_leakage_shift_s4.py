from __future__ import annotations

import pytest

from solverpilot.benchmark.splits import SplitRecord
from solverpilot.intelligence import (
    DistributionShiftError,
    FeatureDefinition,
    LeakageDetectedError,
    ShiftSeverity,
    assess_distribution_shift,
    audit_feature_boundary,
    build_feature_record,
    build_feature_schema,
    fit_distribution_shift_profile,
    fit_training_shift_profile,
)


def schema():
    return build_feature_schema(domain="demo",version="1.0.0",definitions=[
        FeatureDefinition("x","number","ratio","nullable","x",("problem.x",)),
        FeatureDefinition("kind","categorical","category","forbidden","kind",("problem.kind",),allowed_values=("a","b","c")),
        FeatureDefinition("flag","boolean","bool","forbidden","flag",("problem.flag",)),
    ])


def rec(i,x,kind="a",flag=True):
    s=schema(); missing={"x":"missing from source"} if x is None else {}
    return build_feature_record(schema=s,instance_id=i,values={"x":x,"kind":kind,"flag":flag},missingness_reasons=missing,source_content_sha256=(f"{int(i[-1])%10}"*64 if i[-1].isdigit() else "a"*64),extractor_id="demo",extractor_version="1.0.0")


def test_leakage_audit_rejects_solver_runtime_source():
    s=build_feature_schema(domain="demo",version="1.0.0",definitions=[FeatureDefinition("elapsed","number","seconds","forbidden","elapsed",("solve_result.runtime_s",))])
    report=audit_feature_boundary(s); assert not report.passed
    with pytest.raises(LeakageDetectedError): report.require_pass()


def test_leakage_audit_rejects_target_or_winner_name():
    for name in ("winner", "target_speedup", "solver_runtime"):
        s=build_feature_schema(domain="demo",version="1.0.0",definitions=[FeatureDefinition(name,"number","ratio","forbidden","x",("problem.n",))])
        assert not audit_feature_boundary(s).passed


def test_problem_objective_input_is_not_mistaken_for_postsolve_objective():
    s=build_feature_schema(domain="demo",version="1.0.0",definitions=[FeatureDefinition("objective_density","number","ratio","forbidden","pre solve objective coefficients",("problem.objective_coefficients",))])
    assert audit_feature_boundary(s).passed


def test_fit_profile_numeric_statistics_and_immutability():
    s=schema(); rows=[rec("i1",1.0),rec("i2",2.0),rec("i3",3.0)]
    p=fit_distribution_shift_profile(s,rows)
    assert p.statistics["x"]["mean"]==2.0 and p.statistics["x"]["min"]==1.0 and p.statistics["x"]["max"]==3.0
    with pytest.raises(TypeError): p.statistics["x"]["min"]=0


def test_in_distribution_has_no_flags():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1.0),rec("i2",2.0),rec("i3",3.0)])
    a=assess_distribution_shift(p,rec("i4",2.0)); assert a.severity is ShiftSeverity.IN_DISTRIBUTION and not a.flags and not a.shifted


def test_outside_range_is_mild_when_z_not_large():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",0.0),rec("i2",10.0)])
    a=assess_distribution_shift(p,rec("i3",11.0),z_threshold=100)
    assert a.severity is ShiftSeverity.MILD_SHIFT
    assert {f.code for f in a.flags}=={"outside_training_range"}


def test_large_z_is_strong():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",0.0),rec("i2",1.0),rec("i3",2.0)])
    a=assess_distribution_shift(p,rec("i4",20.0),z_threshold=3)
    assert a.severity is ShiftSeverity.STRONG_SHIFT
    assert "large_standardized_distance" in {f.code for f in a.flags}


def test_unseen_category_is_strong():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1,"a"),rec("i2",2,"a")])
    a=assess_distribution_shift(p,rec("i3",1,"b")); assert a.severity is ShiftSeverity.STRONG_SHIFT
    assert any(f.code=="unseen_category" for f in a.flags)


def test_new_missingness_is_mild():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1),rec("i2",2)])
    a=assess_distribution_shift(p,rec("i3",None)); assert a.severity is ShiftSeverity.MILD_SHIFT
    assert any(f.code=="new_missingness" for f in a.flags)


def test_preexisting_missingness_does_not_flag_new_missingness():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",None),rec("i2",2)])
    a=assess_distribution_shift(p,rec("i3",None)); assert not any(f.code=="new_missingness" for f in a.flags)


@pytest.mark.parametrize("z",[0,-1,float("inf"),float("nan"),"3"])
def test_invalid_z_threshold_fails(z):
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1)])
    with pytest.raises(DistributionShiftError): assess_distribution_shift(p,rec("i2",2),z_threshold=z)


def test_schema_mismatch_fails_closed():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1)])
    other=build_feature_schema(domain="other",version="1.0.0",definitions=[FeatureDefinition("y","number","ratio","forbidden","y",("problem.y",))])
    r=build_feature_record(schema=other,instance_id="i2",values={"y":1},source_content_sha256="b"*64,extractor_id="x",extractor_version="1.0.0")
    with pytest.raises(DistributionShiftError): assess_distribution_shift(p,r)


def test_training_profile_uses_only_train_split():
    s=schema(); rows=[rec("i1",1),rec("i2",2),rec("i3",100),rec("i4",200)]
    splits=[SplitRecord("i1","train","g1"),SplitRecord("i2","train","g2"),SplitRecord("i3","validation","g3"),SplitRecord("i4","test","g4")]
    p=fit_training_shift_profile(s,rows,splits)
    assert p.training_instance_ids==("i1","i2")
    assert p.statistics["x"]["max"]==2.0


def test_training_profile_rejects_group_leakage_using_existing_split_validator():
    s=schema(); rows=[rec("i1",1),rec("i2",2)]
    splits=[SplitRecord("i1","train","family"),SplitRecord("i2","test","family")]
    with pytest.raises(DistributionShiftError): fit_training_shift_profile(s,rows,splits)


def test_training_profile_requires_exact_record_coverage():
    s=schema(); rows=[rec("i1",1),rec("i2",2)]
    splits=[SplitRecord("i1","train","g1")]
    with pytest.raises(DistributionShiftError): fit_training_shift_profile(s,rows,splits)


def test_assessment_is_content_addressed_and_stable():
    s=schema(); p=fit_distribution_shift_profile(s,[rec("i1",1),rec("i2",2)])
    a=assess_distribution_shift(p,rec("i3",10)); b=assess_distribution_shift(p,rec("i3",10))
    assert a.assessment_id==b.assessment_id and a.to_dict()==b.to_dict()
