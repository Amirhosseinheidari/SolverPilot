from __future__ import annotations

import numpy as np

import solverpilot
from solverpilot import LinearProblem
from solverpilot.benchmark.splits import SplitRecord, validate_split_records
from solverpilot.intelligence import (
    PROBLEM_FINGERPRINT_FEATURE_SCHEMA,
    ShiftSeverity,
    assess_distribution_shift,
    audit_feature_boundary,
    feature_record_from_problem,
    fit_training_shift_profile,
)


def make_problem(scale: float):
    return LinearProblem.from_data(A=[[1.0*scale,2.0*scale],[0.0,3.0*scale]],c=[1.0,2.0],variable_lower=[0.0,0.0],variable_upper=[10.0,10.0],constraint_lower=[0.0,1.0*scale],constraint_upper=[5.0*scale,1.0*scale])


def test_s4_does_not_change_frozen_top_level_api():
    assert len(solverpilot.__all__)==78
    assert "FeatureSchema" not in solverpilot.__all__
    assert "assess_distribution_shift" not in solverpilot.__all__


def test_existing_problem_inspector_is_single_source_of_features():
    p=make_problem(1.0); r=feature_record_from_problem(p,instance_id="p1")
    from solverpilot import inspect_problem
    f=inspect_problem(p)
    assert r.values["n_variables"]==f.n_variables
    assert r.values["density_a"]==f.density_a
    assert r.values["row_nnz_mean"]==f.row_nnz.mean


def test_existing_split_contract_remains_authoritative():
    rows=[SplitRecord("p1","train","family-a"),SplitRecord("p2","test","family-a")]
    report=validate_split_records(rows,expected_instances=["p1","p2"])
    assert not report.ok and "group leakage" in report.errors[0]


def test_end_to_end_problem_fingerprint_train_profile_and_ood_assessment():
    train=[feature_record_from_problem(make_problem(1.0),instance_id="p1"),feature_record_from_problem(make_problem(1.2),instance_id="p2")]
    candidate=feature_record_from_problem(make_problem(1000.0),instance_id="p3")
    splits=[SplitRecord("p1","train","g1"),SplitRecord("p2","train","g2"),SplitRecord("p3","test","g3")]
    # Profile fitting requires exact record coverage; include candidate record but train only p1/p2.
    profile=fit_training_shift_profile(PROBLEM_FINGERPRINT_FEATURE_SCHEMA,[*train,candidate],splits)
    assessment=assess_distribution_shift(profile,candidate)
    assert profile.training_instance_ids==("p1","p2")
    assert assessment.severity in {ShiftSeverity.MILD_SHIFT,ShiftSeverity.STRONG_SHIFT}


def test_feature_schema_boundary_is_presolve_only():
    report=audit_feature_boundary(PROBLEM_FINGERPRINT_FEATURE_SCHEMA)
    assert report.passed
    assert all(all(src.startswith("problem.") for src in d.source_fields) for d in PROBLEM_FINGERPRINT_FEATURE_SCHEMA.definitions)
