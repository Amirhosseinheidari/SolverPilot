from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

import solverpilot
from solverpilot.io import (
    DataIOConfigurationError,
    DataIOMappingError,
    DataIssue,
    MappingRule,
    ObjectMapping,
    SourceFileProvenance,
    SourceProvenance,
    sha256_json,
)
from solverpilot.io.mapping import set_path
from solverpilot.problem import LinearProblem


def test_top_level_api_remains_frozen() -> None:
    assert len(solverpilot.__all__) == 78
    assert "load_json" not in solverpilot.__all__
    assert "SourceProvenance" not in solverpilot.__all__


def test_mapping_rejects_duplicate_or_prefix_target_collisions() -> None:
    with pytest.raises(DataIOConfigurationError, match="collide"):
        ObjectMapping(
            rules=(
                MappingRule(source_path="a", target_path="x"),
                MappingRule(source_path="b", target_path="x.y"),
            )
        )
    with pytest.raises(DataIOConfigurationError, match="collide"):
        ObjectMapping(
            rules=(MappingRule(source_path="a", target_path="x.y"),),
            constants={"x": 3},
        )


def test_set_path_never_overwrites_leaf_or_subtree() -> None:
    target: dict[str, object] = {"a": 1}
    with pytest.raises(DataIOMappingError, match="collides"):
        set_path(target, "a.b", 2)
    target = {"a": {"b": 1}}
    with pytest.raises(DataIOMappingError, match="overwrite"):
        set_path(target, "a.b", 2)


def test_mapping_configuration_is_detached_and_immutable() -> None:
    source = {"nested": {"items": [1, 2]}}
    config = ObjectMapping(constants=source)
    source["nested"]["items"].append(3)
    assert config.to_dict() == {"kind": "object", "version": "1.0", "identity": False, "constants": {"nested": {"items": [1, 2]}}, "rules": []}
    with pytest.raises(TypeError):
        config.constants["x"] = 1


def test_mapping_rule_constant_and_default_are_detached() -> None:
    default = {"x": [1]}
    constant = {"y": [2]}
    r1 = MappingRule(source_path="a", target_path="a", required=False, default=default)
    r2 = MappingRule(target_path="b", constant=constant)
    default["x"].append(9)
    constant["y"].append(9)
    assert r1.to_dict()["default"] == {"x": [1]}
    assert r2.to_dict()["constant"] == {"y": [2]}


def test_mapping_rejects_cycle_in_constants() -> None:
    cyclic: dict[str, object] = {}
    cyclic["self"] = cyclic
    with pytest.raises(DataIOConfigurationError, match="cycle|nesting"):
        ObjectMapping(constants=cyclic)


def test_mapping_source_path_array_index_is_supported() -> None:
    rule = MappingRule(source_path="items[0].x", target_path="x")
    assert rule.source_path == "items[0].x"


@pytest.mark.parametrize("source", ["a..b", "a.", ".a", "a[-1]", "a[x]"])
def test_mapping_invalid_source_path_is_rejected(source: str) -> None:
    with pytest.raises(DataIOConfigurationError):
        MappingRule(source_path=source, target_path="x")


@pytest.mark.parametrize("target", ["a..b", "a.", ".a", "a[0]"])
def test_mapping_invalid_target_path_is_rejected(target: str) -> None:
    with pytest.raises(DataIOConfigurationError):
        MappingRule(source_path="a", target_path=target)


def test_sha256_json_is_order_independent_for_mapping_keys() -> None:
    assert sha256_json({"b": 2, "a": 1}) == sha256_json({"a": 1, "b": 2})


def _prov() -> SourceProvenance:
    h = "a" * 64
    return SourceProvenance(
        loader="solverpilot.io.json",
        loader_version="1.0",
        source_type="json_file",
        source_name="input.json",
        source_sha256=h.upper(),
        source_size_bytes=5,
        mapping_sha256="b" * 64,
        normalized_payload_sha256="c" * 64,
        policy_sha256="d" * 64,
        record_count=1,
        encoding="utf-8",
        source_files=(SourceFileProvenance("input.json", "E" * 64, 5),),
        metadata={"nested": {"x": [1]}},
    )


def test_provenance_normalizes_hashes_and_is_immutable() -> None:
    p = _prov()
    assert p.source_sha256 == "a" * 64
    assert p.source_files[0].sha256 == "e" * 64
    with pytest.raises(TypeError):
        p.metadata["x"] = 1
    detached = p.to_dict()
    detached["metadata"]["nested"]["x"].append(2)
    assert p.to_dict()["metadata"] == {"nested": {"x": [1]}}


def test_provenance_rejects_invalid_hash_and_unsafe_name() -> None:
    with pytest.raises(DataIOConfigurationError):
        SourceFileProvenance("x", "bad", 1)
    with pytest.raises(DataIOConfigurationError):
        SourceFileProvenance("../x", "a" * 64, 1)


def test_data_issue_contract_is_typed_and_nonempty() -> None:
    issue = DataIssue("INFO", "unused header", "Unused header", path="x.csv")
    assert issue.severity == "info" and issue.code == "unused_header"
    with pytest.raises(DataIOConfigurationError):
        DataIssue("error", "x", "y")
    with pytest.raises(DataIOConfigurationError):
        DataIssue("info", "", "y")


def test_provenance_can_attach_to_problem_without_changing_mathematical_hashes() -> None:
    base = LinearProblem.from_data(
        A=[[1.0]], c=[1.0], variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
    )
    p = _prov()
    enriched = LinearProblem.from_data(
        A=[[1.0]], c=[1.0], variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
        metadata=p.as_problem_metadata(),
    )
    assert enriched.metadata["source_provenance"]["source_sha256"] == "a" * 64
    assert enriched.structural_hash == base.structural_hash
    assert enriched.data_hash == base.data_hash
