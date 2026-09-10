from __future__ import annotations

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, VariableDomain
from solverpilot.intelligence import (
    PROBLEM_FINGERPRINT_FEATURE_SCHEMA,
    FeatureDataType,
    FeatureDefinition,
    FeatureRecord,
    FeatureSchema,
    FeatureSchemaError,
    MissingnessPolicy,
    audit_feature_boundary,
    build_feature_record,
    build_feature_schema,
    feature_record_from_problem,
)


def demo_schema():
    return build_feature_schema(
        domain="demo",
        version="1.2.3",
        definitions=[
            FeatureDefinition("size", "integer", "count", "forbidden", "Problem size.", ("problem.n",), lower_bound=0),
            FeatureDefinition("density", "number", "ratio", "forbidden", "Density.", ("problem.A",), lower_bound=0, upper_bound=1),
            FeatureDefinition("kind", "categorical", "category", "forbidden", "Kind.", ("problem.type",), allowed_values=("a", "b")),
            FeatureDefinition("flag", "boolean", "bool", "nullable", "Optional flag.", ("problem.flag",)),
        ],
    )


def test_schema_is_sorted_and_content_addressed():
    s=demo_schema(); assert s.names==tuple(sorted(s.names)); assert len(s.feature_schema_id)==64
    assert FeatureSchema.from_dict(s.to_dict())==s


def test_schema_id_changes_when_semantics_change():
    a=demo_schema()
    b=build_feature_schema(domain="demo",version="1.2.4",definitions=a.definitions)
    assert a.feature_schema_id!=b.feature_schema_id


@pytest.mark.parametrize("kwargs",[
    dict(name="",data_type="number",unit="ratio",missingness="forbidden",description="x",source_fields=("p",)),
    dict(name="x",data_type="wat",unit="ratio",missingness="forbidden",description="x",source_fields=("p",)),
    dict(name="x",data_type="number",unit="ratio",missingness="forbidden",description="x",source_fields=()),
    dict(name="x",data_type="categorical",unit="category",missingness="forbidden",description="x",source_fields=("p",),allowed_values=()),
    dict(name="x",data_type="number",unit="ratio",missingness="forbidden",description="x",source_fields=("p",),allowed_values=("a",)),
    dict(name="x",data_type="number",unit="ratio",missingness="forbidden",description="x",source_fields=("p",),lower_bound=2,upper_bound=1),
])
def test_invalid_definitions_fail_closed(kwargs):
    with pytest.raises(FeatureSchemaError): FeatureDefinition(**kwargs)


def test_record_is_content_addressed_and_round_trips():
    s=demo_schema()
    r=build_feature_record(schema=s,instance_id="instance-A",values={"size":3,"density":0.25,"kind":"a","flag":None},missingness_reasons={"flag":"not available"},source_content_sha256="a"*64,extractor_id="demo.extractor",extractor_version="1.0.0")
    assert len(r.record_id)==64
    assert FeatureRecord.from_dict(r.to_dict(),schema=s)==r
    with pytest.raises(TypeError): r.values["size"]=4


def test_record_id_changes_with_feature_value_or_source_hash():
    s=demo_schema()
    kw=dict(schema=s,instance_id="i",missingness_reasons={},extractor_id="e",extractor_version="1.0.0")
    a=build_feature_record(values={"size":1,"density":0.2,"kind":"a","flag":True},source_content_sha256="a"*64,**kw)
    b=build_feature_record(values={"size":2,"density":0.2,"kind":"a","flag":True},source_content_sha256="a"*64,**kw)
    c=build_feature_record(values={"size":1,"density":0.2,"kind":"a","flag":True},source_content_sha256="b"*64,**kw)
    assert len({a.record_id,b.record_id,c.record_id})==3


@pytest.mark.parametrize("values,reasons",[
    ({"size":1,"density":0.2,"kind":"a"},{}),
    ({"size":1,"density":0.2,"kind":"a","flag":None},{}),
    ({"size":1,"density":0.2,"kind":"a","flag":True},{"flag":"wrong"}),
    ({"size":-1,"density":0.2,"kind":"a","flag":True},{}),
    ({"size":1,"density":1.2,"kind":"a","flag":True},{}),
    ({"size":1,"density":0.2,"kind":"c","flag":True},{}),
])
def test_record_validation_fails_closed(values,reasons):
    with pytest.raises(FeatureSchemaError):
        build_feature_record(schema=demo_schema(),instance_id="i",values=values,missingness_reasons=reasons,source_content_sha256="a"*64,extractor_id="e",extractor_version="1.0.0")


def _lp():
    return LinearProblem.from_data(A=[[1.0,0.0],[2.0,3.0]],c=[1.0,4.0],variable_lower=[0,0],variable_upper=[10,10],constraint_lower=[0,1],constraint_upper=[5,5])


def test_problem_fingerprint_schema_is_leakage_clean():
    report=audit_feature_boundary(PROBLEM_FINGERPRINT_FEATURE_SCHEMA)
    assert report.passed, report.findings


def test_problem_fingerprint_record_uses_existing_inspector_and_problem_data_hash():
    p=_lp(); r=feature_record_from_problem(p)
    assert r.instance_id==p.data_hash and r.source_content_sha256==p.data_hash
    assert r.values["n_variables"]==2 and r.values["n_constraints"]==2
    assert r.values["problem_class"]=="lp"
    assert "structural_hash" not in r.values


def test_instance_id_can_be_benchmark_name_without_changing_source_hash():
    p=_lp(); r=feature_record_from_problem(p,instance_id="aflow40b.mps.gz")
    assert r.instance_id=="aflow40b.mps.gz"
    assert r.source_content_sha256==p.data_hash


def test_milp_fingerprint_record():
    p=LinearProblem.from_data(A=[[1,1]],c=[1,2],variable_lower=[0,0],variable_upper=[1,5],constraint_lower=[1],constraint_upper=[3],domains=[VariableDomain.BINARY,VariableDomain.INTEGER])
    r=feature_record_from_problem(p)
    assert r.values["problem_class"]=="milp"
    assert r.values["integer_fraction"]==1.0


def test_qp_fingerprint_record_has_quadratic_features():
    q=QuadraticProblem.from_data(P=[[2.0,0.0],[0.0,4.0]],A=[[1,1]],q=[1,1],variable_lower=[0,0],variable_upper=[10,10],constraint_lower=[0],constraint_upper=[5])
    r=feature_record_from_problem(q)
    assert r.values["problem_class"]=="convex_qp"
    assert r.values["quadratic_nnz"]==2
    assert r.values["convexity_status"]=="confirmed"


def test_lp_marks_qp_only_features_missing_with_reasons():
    r=feature_record_from_problem(_lp())
    for name in ("quadratic_nnz","quadratic_density","convexity_status"):
        assert r.values[name] is None and name in r.missingness_reasons
