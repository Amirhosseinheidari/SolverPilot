from __future__ import annotations

import os
from pathlib import Path

import pytest

from solverpilot.io import (
    DataIOConfigurationError,
    DataIOMappingError,
    DataIOParseError,
    DataIOSourceError,
    LoadPolicy,
    MappingRule,
    ObjectMapping,
    load_json,
)


def test_json_identity_is_immutable_and_records_provenance(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"b":2,"a":[1,true]}', encoding="utf-8")
    result = load_json(path)
    assert list(result.payload.keys()) == ["a", "b"]
    assert result.provenance.source_type == "json_file"
    assert result.provenance.source_name == "input.json"
    assert result.provenance.source_files[0].name == "input.json"
    assert result.provenance.normalized_payload_sha256
    with pytest.raises(TypeError):
        result.payload["x"] = 1
    mutable = result.mutable_payload()
    mutable["x"] = 1
    assert "x" not in result.payload


def test_json_mapping_nested_paths_types_defaults_and_constants(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"source":{"value":" 7 ","flag":"yes"},"items":[{"x":"1.25"}]}')
    mapping = ObjectMapping(
        rules=(
            MappingRule(source_path="source.value", target_path="n", value_type="integer"),
            MappingRule(source_path="source.flag", target_path="flag", value_type="boolean"),
            MappingRule(source_path="items[0].x", target_path="stats.x", value_type="number"),
            MappingRule(source_path="missing", target_path="optional", required=False),
            MappingRule(source_path="absent", target_path="defaulted", default="z"),
        ),
        constants={"kind": "sample"},
    )
    result = load_json(path, mapping=mapping)
    assert result.mutable_payload() == {
        "kind": "sample",
        "n": 7,
        "flag": True,
        "stats": {"x": 1.25},
        "optional": None,
        "defaulted": "z",
    }


def test_json_rejects_duplicate_keys(tmp_path: Path) -> None:
    path = tmp_path / "dup.json"
    path.write_text('{"secret":"VALUE_ALPHA_123","secret":"VALUE_BETA_456"}')
    with pytest.raises(DataIOParseError, match="duplicate") as exc:
        load_json(path)
    assert "VALUE_ALPHA_123" not in str(exc.value)
    assert "VALUE_BETA_456" not in str(exc.value)


@pytest.mark.parametrize("constant", ["NaN", "Infinity", "-Infinity"])
def test_json_rejects_nonfinite_constants(tmp_path: Path, constant: str) -> None:
    path = tmp_path / "bad.json"
    path.write_text('{"x":' + constant + "}")
    with pytest.raises(DataIOParseError, match="non-finite"):
        load_json(path)


def test_json_strict_policy_rejects_decimal_precision_loss(tmp_path: Path) -> None:
    path = tmp_path / "number.json"
    path.write_text('{"x":0.10000000000000001}')
    with pytest.raises(DataIOParseError, match="binary64"):
        load_json(path)


def test_json_relaxed_policy_accepts_finite_binary64_rounding(tmp_path: Path) -> None:
    path = tmp_path / "number.json"
    path.write_text('{"x":0.10000000000000001}')
    result = load_json(path, policy=LoadPolicy(strict_numbers=False))
    assert result.payload["x"] == 0.1
    assert result.provenance.metadata["strict_numbers"] is False


@pytest.mark.parametrize("value", ["0.1", "1.25", "1e-3", "-12.5"])
def test_json_strict_policy_accepts_common_roundtrip_safe_values(tmp_path: Path, value: str) -> None:
    path = tmp_path / "number.json"
    path.write_text('{"x":' + value + "}")
    result = load_json(path)
    assert isinstance(result.payload["x"], float)


def test_json_rejects_binary64_underflow(tmp_path: Path) -> None:
    path = tmp_path / "number.json"
    path.write_text('{"x":1e-4000}')
    with pytest.raises(DataIOParseError, match="binary64"):
        load_json(path)


def test_json_rejects_excessive_integer_digits(tmp_path: Path) -> None:
    path = tmp_path / "number.json"
    path.write_text('{"x":' + "1" * 1001 + "}")
    with pytest.raises(DataIOParseError, match="1000"):
        load_json(path)


def test_json_rejects_depth_over_limit(tmp_path: Path) -> None:
    path = tmp_path / "deep.json"
    path.write_text("[[[0]]]", encoding="utf-8")
    with pytest.raises(DataIOParseError, match="nesting"):
        load_json(path, policy=LoadPolicy(max_json_depth=1))


def test_json_rejects_wrong_encoding(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_bytes('{"x":"é"}'.encode("latin-1"))
    with pytest.raises(DataIOParseError, match="decode"):
        load_json(path, policy=LoadPolicy(encoding="utf-8"))


def test_json_rejects_unknown_encoding(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"x":1}')
    with pytest.raises(DataIOSourceError, match="encoding"):
        load_json(path, policy=LoadPolicy(encoding="definitely-not-an-encoding"))


def test_json_size_limit_accepts_exact_and_rejects_one_extra(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    raw = b'{"x":1}'
    path.write_bytes(raw)
    assert load_json(path, policy=LoadPolicy(max_source_bytes=len(raw))).payload["x"] == 1
    with pytest.raises(DataIOSourceError, match="maximum size"):
        load_json(path, policy=LoadPolicy(max_source_bytes=len(raw) - 1))


def test_json_rejects_symlinked_source_by_default(tmp_path: Path) -> None:
    target = tmp_path / "target.json"
    target.write_text('{"x":1}')
    link = tmp_path / "link.json"
    try:
        link.symlink_to(target)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(DataIOSourceError, match="Symbolic"):
        load_json(link)
    assert load_json(link, policy=LoadPolicy(reject_symlinks=False)).payload["x"] == 1


def test_json_rejects_symlinked_parent_by_default(tmp_path: Path) -> None:
    real = tmp_path / "real"
    real.mkdir()
    (real / "x.json").write_text('{"x":1}')
    link = tmp_path / "linked"
    try:
        link.symlink_to(real, target_is_directory=True)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    with pytest.raises(DataIOSourceError, match="Symbolic"):
        load_json(link / "x.json")


def test_json_missing_required_mapping_does_not_echo_source_content(tmp_path: Path) -> None:
    path = tmp_path / "input.json"
    path.write_text('{"password":"do-not-echo"}')
    mapping = ObjectMapping(rules=(MappingRule(source_path="missing", target_path="x"),))
    with pytest.raises(DataIOMappingError) as exc:
        load_json(path, mapping=mapping)
    assert "do-not-echo" not in str(exc.value)


def test_json_policy_validation_is_fail_closed() -> None:
    with pytest.raises(DataIOConfigurationError):
        LoadPolicy(reject_symlinks=1)  # type: ignore[arg-type]
    with pytest.raises(DataIOConfigurationError):
        LoadPolicy(strict_numbers=1)  # type: ignore[arg-type]
    with pytest.raises(DataIOConfigurationError):
        LoadPolicy(delimiter='"')
