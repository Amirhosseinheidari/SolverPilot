from __future__ import annotations

import os
from pathlib import Path

import pytest

from solverpilot.io import (
    CsvMapping,
    CsvTable,
    DataIOConfigurationError,
    DataIOMappingError,
    DataIOParseError,
    DataIOSourceError,
    LoadPolicy,
    MappingRule,
    load_csv,
)


def _table(*, file_name: str | None = None, target_path: str = "rows", delimiter: str | None = None) -> CsvTable:
    return CsvTable(
        target_path=target_path,
        file_name=file_name,
        delimiter=delimiter,
        fields=(
            MappingRule(source_path="id", target_path="id", value_type="string"),
            MappingRule(source_path="value", target_path="value", value_type="number"),
        ),
    )


def test_csv_loads_single_file_and_records_provenance(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value\nA,1.25\nB,2.5\n", encoding="utf-8")
    result = load_csv(path, mapping=CsvMapping(tables=(_table(),)))
    assert result.mutable_payload() == {"rows": [{"id": "A", "value": 1.25}, {"id": "B", "value": 2.5}]}
    assert result.provenance.source_type == "csv_file"
    assert result.provenance.record_count == 2
    assert result.provenance.source_files[0].name == "x.csv"


def test_csv_empty_target_path_returns_record_list(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value\nA,1.25\n")
    result = load_csv(path, mapping=CsvMapping(tables=(_table(target_path=""),)))
    assert result.mutable_payload() == [{"id": "A", "value": 1.25}]


def test_csv_semicolon_and_utf8_bom(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_bytes("id;value\nÅ;1.25\n".encode("utf-8-sig"))
    result = load_csv(
        path,
        mapping=CsvMapping(tables=(_table(delimiter=";"),)),
        policy=LoadPolicy(encoding="utf-8-sig"),
    )
    assert result.mutable_payload()["rows"][0]["id"] == "Å"


def test_csv_rejects_duplicate_headers(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,id,value\nA,B,1.0\n")
    with pytest.raises(DataIOParseError, match="unique"):
        load_csv(path, mapping=CsvMapping(tables=(_table(),)))


def test_csv_rejects_missing_required_header(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id\nA\n")
    with pytest.raises(DataIOParseError, match="missing required"):
        load_csv(path, mapping=CsvMapping(tables=(_table(),)))


def test_csv_rejects_inconsistent_width_and_blank_rows(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value\nA,1,extra\n")
    with pytest.raises(DataIOParseError, match="field count"):
        load_csv(path, mapping=CsvMapping(tables=(_table(),)))
    path.write_text("id,value\n\n")
    with pytest.raises(DataIOParseError, match="blank"):
        load_csv(path, mapping=CsvMapping(tables=(_table(),)))


def test_csv_strict_and_relaxed_number_policy(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value\nA,0.10000000000000001\n")
    mapping = CsvMapping(tables=(_table(),))
    with pytest.raises(DataIOMappingError, match="precision"):
        load_csv(path, mapping=mapping)
    result = load_csv(path, mapping=mapping, policy=LoadPolicy(strict_numbers=False))
    assert result.mutable_payload()["rows"][0]["value"] == 0.1


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "-Infinity", "1e-4000"])
def test_csv_rejects_nonfinite_or_underflow_numeric_text(tmp_path: Path, bad: str) -> None:
    path = tmp_path / "x.csv"
    path.write_text(f"id,value\nA,{bad}\n")
    with pytest.raises(DataIOMappingError):
        load_csv(path, mapping=CsvMapping(tables=(_table(),)))


def test_csv_reports_unused_headers_as_info(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value,unused\nA,1.25,z\n")
    result = load_csv(path, mapping=CsvMapping(tables=(_table(),)))
    assert len(result.issues) == 1
    assert result.issues[0].code == "unused_csv_headers"


def test_csv_bundle_is_declared_not_discovered(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "a.csv").write_text("id,value\nA,1.0\n")
    (bundle / "b.csv").write_text("id,value\nB,2.0\n")
    (bundle / "ignored.csv").write_text("id,value\nX,99.0\n")
    mapping = CsvMapping(
        tables=(
            _table(file_name="a.csv", target_path="a"),
            _table(file_name="b.csv", target_path="b"),
        ),
        constants={"kind": "bundle"},
    )
    result = load_csv(bundle, mapping=mapping)
    payload = result.mutable_payload()
    assert payload["a"][0]["id"] == "A" and payload["b"][0]["id"] == "B"
    assert "ignored" not in payload
    assert [x.name for x in result.provenance.source_files] == ["a.csv", "b.csv"]


def test_csv_bundle_rejects_path_traversal_at_configuration_time() -> None:
    with pytest.raises(DataIOConfigurationError, match="safe relative"):
        _table(file_name="../secret.csv")
    with pytest.raises(DataIOConfigurationError, match="safe relative"):
        _table(file_name="C:/secret.csv")


def test_csv_directory_requires_file_names(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    (bundle / "x.csv").write_text("id,value\nA,1.0\n")
    with pytest.raises(DataIOConfigurationError, match="declare file_name"):
        load_csv(bundle, mapping=CsvMapping(tables=(_table(),)))


def test_csv_bundle_rejects_symlinked_child(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    outside = tmp_path / "outside.csv"
    outside.write_text("id,value\nA,1.0\n")
    link = bundle / "x.csv"
    try:
        link.symlink_to(outside)
    except (OSError, NotImplementedError):
        pytest.skip("symlinks unavailable")
    mapping = CsvMapping(tables=(_table(file_name="x.csv"),))
    with pytest.raises(DataIOSourceError, match="Symbolic"):
        load_csv(bundle, mapping=mapping)


def test_csv_combined_size_limit_is_enforced(tmp_path: Path) -> None:
    bundle = tmp_path / "bundle"
    bundle.mkdir()
    a = b"id,value\nA,1.0\n"
    b = b"id,value\nB,2.0\n"
    (bundle / "a.csv").write_bytes(a)
    (bundle / "b.csv").write_bytes(b)
    mapping = CsvMapping(tables=(_table(file_name="a.csv", target_path="a"), _table(file_name="b.csv", target_path="b")))
    with pytest.raises(DataIOSourceError, match="maximum|size"):
        load_csv(bundle, mapping=mapping, policy=LoadPolicy(max_source_bytes=len(a) + len(b) - 1))


def test_csv_rejects_duplicate_and_prefix_targets() -> None:
    with pytest.raises(DataIOConfigurationError, match="collide"):
        CsvMapping(tables=(_table(file_name="a.csv", target_path="x"), _table(file_name="b.csv", target_path="x.y")))


def test_csv_single_file_rejects_multiple_table_mappings(tmp_path: Path) -> None:
    path = tmp_path / "x.csv"
    path.write_text("id,value\nA,1.0\n")
    mapping = CsvMapping(tables=(_table(target_path="a"), _table(target_path="b")))
    with pytest.raises(DataIOConfigurationError, match="exactly one"):
        load_csv(path, mapping=mapping)
