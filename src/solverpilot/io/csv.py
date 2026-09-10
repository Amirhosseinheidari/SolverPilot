"""Safe deterministic local CSV and explicitly declared CSV-bundle ingestion."""

from __future__ import annotations

import csv
import io
from pathlib import Path
from typing import Any

from .errors import (
    DataIOConfigurationError,
    DataIOMappingError,
    DataIOParseError,
    DataIOSourceError,
)
from .hashing import canonicalize_json_value, deep_thaw_json, sha256_bytes, sha256_json
from .mapping import apply_rules, set_path
from .model import CsvMapping, CsvTable, DataIssue, LoadPolicy
from .provenance import SourceFileProvenance, SourceProvenance
from .result import LoadedData
from .source import read_limited_bytes, safe_bundle_child, validate_local_path

_CSV_LOADER_VERSION = "1.0"


def _parse_table(
    raw_bytes: bytes,
    *,
    table: CsvTable,
    encoding: str,
    delimiter: str,
    policy: LoadPolicy,
) -> tuple[list[tuple[int, dict[str, str]]], tuple[str, ...]]:
    try:
        text = raw_bytes.decode(encoding)
    except LookupError as exc:
        raise DataIOSourceError("Declared CSV encoding is not supported.") from exc
    except UnicodeDecodeError as exc:
        raise DataIOParseError("CSV source cannot be decoded with the declared encoding.") from exc

    try:
        reader = csv.reader(io.StringIO(text, newline=""), delimiter=delimiter, strict=True)
        try:
            raw_headers = next(reader)
        except StopIteration as exc:
            raise DataIOParseError("CSV source is missing a header row.") from exc
        headers = tuple(str(header).strip() for header in raw_headers)
        if any(not header for header in headers):
            raise DataIOParseError("CSV header names must be non-empty.")
        if len(headers) != len(set(headers)):
            raise DataIOParseError("CSV header names must be unique.")

        required_headers = set(table.required_headers)
        required_headers.update(
            rule.source_path
            for rule in table.fields
            if rule.source_path is not None and rule.required and not rule.has_constant
        )
        missing = sorted(str(item) for item in required_headers.difference(headers))
        if missing:
            raise DataIOParseError(f"CSV source is missing required headers: {', '.join(missing)}.")

        rows: list[tuple[int, dict[str, str]]] = []
        for raw_row in reader:
            row_number = reader.line_num
            if not raw_row:
                raise DataIOParseError(f"CSV row {row_number} is blank; blank data rows are not allowed.")
            if len(raw_row) != len(headers):
                raise DataIOParseError(
                    f"CSV row {row_number} has a different field count than the header."
                )
            rows.append((row_number, dict(zip(headers, map(str, raw_row)))))
        return rows, headers
    except DataIOParseError:
        raise
    except (csv.Error, ValueError) as exc:
        raise DataIOParseError("CSV source or delimiter configuration is invalid.") from exc


def _resolve_table_path(root: Path, table: CsvTable, policy: LoadPolicy) -> Path:
    if root.is_file():
        if table.file_name not in {None, root.name}:
            raise DataIOConfigurationError(
                "Single-file CSV mapping file_name must be omitted or match the source name."
            )
        return root
    if table.file_name is None:
        raise DataIOConfigurationError("Every CSV table in a directory bundle must declare file_name.")
    return safe_bundle_child(root, table.file_name, reject_symlinks=policy.reject_symlinks)


def load_csv(
    source: str | Path,
    *,
    mapping: CsvMapping,
    policy: LoadPolicy | None = None,
) -> LoadedData:
    """Load one CSV file or an explicitly declared local CSV directory bundle."""
    policy = LoadPolicy() if policy is None else policy
    if not isinstance(mapping, CsvMapping):
        raise DataIOConfigurationError("mapping must be CsvMapping.")
    if not isinstance(policy, LoadPolicy):
        raise DataIOConfigurationError("policy must be LoadPolicy.")

    root = validate_local_path(
        source,
        expect_directory=None,
        reject_symlinks=policy.reject_symlinks,
    )
    if root.is_dir() and any(table.file_name is None for table in mapping.tables):
        raise DataIOConfigurationError("Every CSV table in a directory bundle must declare file_name.")
    if root.is_file() and len(mapping.tables) != 1:
        raise DataIOConfigurationError("A single CSV file source requires exactly one table mapping.")

    normalized: Any = canonicalize_json_value(deep_thaw_json(mapping.constants))
    if not isinstance(normalized, dict):
        raise DataIOConfigurationError("CSV constants must form an object.")

    source_files: list[SourceFileProvenance] = []
    issues: list[DataIssue] = []
    total_size = 0
    total_records = 0
    table_metadata: list[dict[str, Any]] = []

    for table in mapping.tables:
        table_path = _resolve_table_path(root, table, policy)
        source_name = table_path.name if root.is_file() else str(table.file_name)
        remaining = policy.max_source_bytes - total_size
        if remaining <= 0:
            raise DataIOSourceError("Combined CSV bundle size exceeds the configured maximum.")
        raw_bytes = read_limited_bytes(table_path, max_bytes=remaining)
        total_size += len(raw_bytes)
        encoding = table.encoding or policy.encoding
        delimiter = table.delimiter or policy.delimiter
        rows, headers = _parse_table(
            raw_bytes,
            table=table,
            encoding=encoding,
            delimiter=delimiter,
            policy=policy,
        )
        if len(rows) < table.min_rows:
            raise DataIOParseError(
                f"CSV table {source_name!r} has {len(rows)} data rows; minimum required is {table.min_rows}."
            )
        used_headers = {rule.source_path for rule in table.fields if rule.source_path is not None}
        unused = tuple(header for header in headers if header not in used_headers)
        if unused:
            issues.append(
                DataIssue(
                    severity="info",
                    code="unused_csv_headers",
                    message=f"CSV table {source_name!r} contains headers not used by the explicit mapping.",
                    path=source_name,
                )
            )

        records: list[dict[str, Any]] = []
        for row_number, row in rows:
            try:
                records.append(
                    apply_rules(
                        row,
                        table.fields,
                        strict_numbers=policy.strict_numbers,
                        max_json_depth=policy.max_json_depth,
                        row_number=row_number,
                    )
                )
            except DataIOMappingError as exc:
                raise DataIOMappingError(f"CSV mapping failed in {source_name!r}: {exc}") from exc

        if table.target_path:
            set_path(normalized, table.target_path, records)
        else:
            if len(mapping.tables) != 1 or mapping.constants:
                raise DataIOConfigurationError(
                    "An empty CSV target_path is only valid for one table with no top-level constants."
                )
            normalized = records

        file_hash = sha256_bytes(raw_bytes)
        source_files.append(
            SourceFileProvenance(name=source_name, sha256=file_hash, size_bytes=len(raw_bytes))
        )
        total_records += len(records)
        table_metadata.append(
            {
                "source_name": source_name,
                "target_path": table.target_path,
                "row_count": len(records),
                "encoding": encoding,
                "delimiter": delimiter,
            }
        )

    evidence = tuple(sorted(source_files, key=lambda item: item.name.casefold()))
    combined_source_hash = (
        evidence[0].sha256
        if root.is_file() and len(evidence) == 1
        else sha256_json([item.to_dict() for item in evidence])
    )
    provenance = SourceProvenance(
        loader="solverpilot.io.csv",
        loader_version=_CSV_LOADER_VERSION,
        source_type="csv_bundle" if root.is_dir() else "csv_file",
        source_name=root.name,
        source_sha256=combined_source_hash,
        source_size_bytes=total_size,
        mapping_sha256=sha256_json(mapping.to_dict()),
        normalized_payload_sha256=sha256_json(normalized),
        policy_sha256=sha256_json(policy.to_dict()),
        record_count=total_records,
        encoding=policy.encoding,
        delimiter=policy.delimiter,
        source_files=evidence,
        metadata={
            "mapping_version": mapping.version,
            "reject_symlinks": policy.reject_symlinks,
            "strict_numbers": policy.strict_numbers,
            "bundle_name_policy": "normalized_posix_casefold_unique_v1",
            "tables": table_metadata,
        },
    )
    return LoadedData(
        payload=normalized,
        provenance=provenance,
        issues=tuple(issues),
        metadata={"mapping_kind": "csv", "table_count": len(mapping.tables)},
    )
