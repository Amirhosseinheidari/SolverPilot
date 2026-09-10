from __future__ import annotations

import json

from solverpilot.benchmark.splits import load_split_records, validate_split_file, validate_split_records, SplitRecord


def test_group_aware_split_accepts_disjoint_groups():
    records=(SplitRecord("a1","train","a"),SplitRecord("a2","train","a"),SplitRecord("b1","test","b"))
    report=validate_split_records(records,expected_instances=["a1","a2","b1"])
    assert report.ok
    assert report.groups==2


def test_group_aware_split_detects_leakage():
    records=(SplitRecord("a1","train","family-a"),SplitRecord("a2","test","family-a"))
    report=validate_split_records(records)
    assert not report.ok
    assert any("group leakage" in e for e in report.errors)


def test_split_exact_coverage_is_enforced():
    report=validate_split_records((SplitRecord("a","train"),),expected_instances=["a","b"])
    assert not report.ok
    assert any("missing split" in e for e in report.errors)


def test_split_loader_and_hash_are_artifact_specific(tmp_path):
    p=tmp_path/"split.json"; p.write_text(json.dumps({"a":{"split":"train","group":"g1"},"b":"test"}))
    records=load_split_records(p)
    assert len(records)==2
    first=validate_split_file(p)
    p.write_text(json.dumps({"a":{"split":"train","group":"g1"},"b":"validation"}))
    second=validate_split_file(p)
    assert first.sha256 != second.sha256
