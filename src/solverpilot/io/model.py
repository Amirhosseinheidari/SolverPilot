"""Small immutable contracts for deterministic local data ingestion."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import DataIOConfigurationError
from .hashing import deep_freeze_json, deep_thaw_json

_MISSING = object()
_ALLOWED_VALUE_TYPES = frozenset({"any", "string", "integer", "number", "boolean", "json"})
_MAPPING_VERSION = "1.0"
_SOURCE_PATH_RE = re.compile(r"(?:[^.\[\]]+|\[[0-9]+\])(?:\.[^.\[\]]+|\[[0-9]+\])*")
_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _require_bool(value: Any, *, field_name: str) -> bool:
    if type(value) is not bool:
        raise DataIOConfigurationError(f"{field_name} must be a boolean.")
    return value


def _clean_target_path(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DataIOConfigurationError(f"{field_name} must be a string.")
    path = value.strip()
    if not path:
        if allow_empty:
            return ""
        raise DataIOConfigurationError(f"{field_name} must be non-empty.")
    parts = path.split(".")
    if any(not part or "[" in part or "]" in part for part in parts):
        raise DataIOConfigurationError(f"{field_name} contains an invalid target path.")
    return path


def _clean_source_path(value: str, *, field_name: str, allow_empty: bool = False) -> str:
    if not isinstance(value, str):
        raise DataIOConfigurationError(f"{field_name} must be a string.")
    path = value.strip()
    if not path:
        if allow_empty:
            return ""
        raise DataIOConfigurationError(f"{field_name} must be non-empty.")
    if _SOURCE_PATH_RE.fullmatch(path) is None:
        raise DataIOConfigurationError(f"{field_name} contains an invalid source path.")
    return path


def _validate_delimiter(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or len(value) != 1:
        raise DataIOConfigurationError(f"{field_name} must be exactly one character.")
    if value in {"\r", "\n", "\x00", '"'}:
        raise DataIOConfigurationError(f"{field_name} cannot be newline, NUL, or quote.")
    if ord(value) < 32 and value != "\t":
        raise DataIOConfigurationError(f"{field_name} cannot be an unsupported control character.")
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataIOConfigurationError(f"{field_name} contains an invalid Unicode surrogate.") from exc
    return value


def _normalize_bundle_name(value: str | None) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        raise DataIOConfigurationError("file_name must be a string.")
    raw = value.strip().replace("\\", "/")
    if not raw:
        return None
    posix = PurePosixPath(raw)
    if posix.is_absolute() or any(part in {"", ".", ".."} or ":" in part for part in posix.parts):
        raise DataIOConfigurationError("file_name must be a safe relative path.")
    return posix.as_posix()


def _target_parts(path: str) -> tuple[str, ...]:
    return tuple(path.split(".")) if path else ()


def _paths_collide(left: tuple[str, ...], right: tuple[str, ...]) -> bool:
    shortest = min(len(left), len(right))
    return left[:shortest] == right[:shortest]


def _constant_paths(
    value: Mapping[str, Any],
    prefix: tuple[str, ...] = (),
    *,
    _active: set[int] | None = None,
) -> tuple[tuple[str, ...], ...]:
    active = set() if _active is None else _active
    identity = id(value)
    if identity in active or len(prefix) > 256:
        raise DataIOConfigurationError(
            "Mapping constants exceed supported nesting or contain a cycle."
        )
    active.add(identity)
    try:
        paths: list[tuple[str, ...]] = []
        for key, item in value.items():
            if not isinstance(key, str) or not key:
                raise DataIOConfigurationError("Constant mapping keys must be non-empty strings.")
            path = prefix + (key,)
            paths.append(path)
            if isinstance(item, Mapping):
                paths.extend(_constant_paths(item, path, _active=active))
        return tuple(paths)
    finally:
        active.remove(identity)


def _validate_target_tree(paths: Sequence[str], *, constants: Mapping[str, Any], label: str) -> None:
    named = [(path, _target_parts(path)) for path in paths if path]
    for i, (left_name, left) in enumerate(named):
        for right_name, right in named[i + 1 :]:
            if _paths_collide(left, right):
                raise DataIOConfigurationError(
                    f"{label} target paths {left_name!r} and {right_name!r} collide."
                )
    constant_paths = _constant_paths(constants)
    for path_name, path in named:
        for constant_path in constant_paths:
            if _paths_collide(path, constant_path):
                raise DataIOConfigurationError(
                    f"{label} target path {path_name!r} collides with constant path "
                    f"{'.'.join(constant_path)!r}."
                )


@dataclass(frozen=True, slots=True)
class LoadPolicy:
    """Local-source policy shared by JSON and CSV ingestion."""

    encoding: str = "utf-8-sig"
    max_source_bytes: int = 16 * 1024 * 1024
    reject_symlinks: bool = True
    strict_numbers: bool = True
    max_json_depth: int = 256
    delimiter: str = ","

    def __post_init__(self) -> None:
        if not isinstance(self.encoding, str) or not self.encoding.strip():
            raise DataIOConfigurationError("encoding must be a non-empty string.")
        if type(self.max_source_bytes) is not int or self.max_source_bytes <= 0:
            raise DataIOConfigurationError("max_source_bytes must be a positive integer.")
        if type(self.max_json_depth) is not int or self.max_json_depth < 0:
            raise DataIOConfigurationError("max_json_depth must be a non-negative integer.")
        object.__setattr__(self, "reject_symlinks", _require_bool(self.reject_symlinks, field_name="reject_symlinks"))
        object.__setattr__(self, "strict_numbers", _require_bool(self.strict_numbers, field_name="strict_numbers"))
        object.__setattr__(self, "encoding", self.encoding.strip())
        object.__setattr__(self, "delimiter", _validate_delimiter(self.delimiter, field_name="delimiter"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "encoding": self.encoding,
            "max_source_bytes": self.max_source_bytes,
            "reject_symlinks": self.reject_symlinks,
            "strict_numbers": self.strict_numbers,
            "max_json_depth": self.max_json_depth,
            "delimiter": self.delimiter,
        }


@dataclass(frozen=True, slots=True)
class MappingRule:
    """One explicit source-to-target normalization rule."""

    target_path: str
    source_path: str | None = None
    value_type: str = "any"
    required: bool = True
    trim: bool = True
    null_values: tuple[str, ...] = ("", "null")
    default: Any = field(default=_MISSING, repr=False)
    constant: Any = field(default=_MISSING, repr=False)

    def __post_init__(self) -> None:
        target_path = _clean_target_path(self.target_path, field_name="target_path")
        source_path = None if self.source_path is None else _clean_source_path(
            self.source_path, field_name="source_path", allow_empty=True
        )
        if not isinstance(self.value_type, str):
            raise DataIOConfigurationError("value_type must be a string.")
        value_type = self.value_type.strip().lower() or "any"
        if value_type not in _ALLOWED_VALUE_TYPES:
            raise DataIOConfigurationError(f"Unsupported value_type {value_type!r}.")
        if self.constant is not _MISSING and source_path is not None:
            raise DataIOConfigurationError("A mapping rule cannot declare source_path and constant.")
        if self.constant is _MISSING and source_path is None:
            raise DataIOConfigurationError("A mapping rule must declare source_path or constant.")
        if not isinstance(self.null_values, Sequence) or isinstance(self.null_values, (str, bytes, bytearray)):
            raise DataIOConfigurationError("null_values must be a sequence of strings.")
        if any(not isinstance(item, str) for item in self.null_values):
            raise DataIOConfigurationError("null_values entries must be strings.")
        object.__setattr__(self, "target_path", target_path)
        object.__setattr__(self, "source_path", source_path)
        object.__setattr__(self, "value_type", value_type)
        object.__setattr__(self, "required", _require_bool(self.required, field_name="required"))
        object.__setattr__(self, "trim", _require_bool(self.trim, field_name="trim"))
        object.__setattr__(self, "null_values", tuple(item.strip().lower() for item in self.null_values))
        if self.default is not _MISSING:
            object.__setattr__(self, "default", deep_freeze_json(self.default))
        if self.constant is not _MISSING:
            object.__setattr__(self, "constant", deep_freeze_json(self.constant))

    @property
    def has_default(self) -> bool:
        return self.default is not _MISSING

    @property
    def has_constant(self) -> bool:
        return self.constant is not _MISSING

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "target_path": self.target_path,
            "value_type": self.value_type,
            "required": self.required,
            "trim": self.trim,
            "null_values": list(self.null_values),
        }
        if self.source_path is not None:
            data["source_path"] = self.source_path
        if self.has_default:
            data["default"] = deep_thaw_json(self.default)
        if self.has_constant:
            data["constant"] = deep_thaw_json(self.constant)
        return data


@dataclass(frozen=True, slots=True)
class ObjectMapping:
    """Explicit deterministic mapping for JSON-compatible input."""

    identity: bool = False
    rules: tuple[MappingRule, ...] = ()
    constants: Mapping[str, Any] = field(default_factory=dict)
    version: str = _MAPPING_VERSION

    def __post_init__(self) -> None:
        if type(self.identity) is not bool:
            raise DataIOConfigurationError("identity must be a boolean.")
        if not isinstance(self.rules, Sequence) or isinstance(self.rules, (str, bytes, bytearray)):
            raise DataIOConfigurationError("rules must be a sequence.")
        rules = tuple(self.rules)
        if any(not isinstance(rule, MappingRule) for rule in rules):
            raise DataIOConfigurationError("rules must contain MappingRule values.")
        if not isinstance(self.constants, Mapping):
            raise DataIOConfigurationError("constants must be a mapping.")
        if self.version != _MAPPING_VERSION:
            raise DataIOConfigurationError(f"Unsupported mapping version {self.version!r}.")
        if self.identity and (rules or self.constants):
            raise DataIOConfigurationError("Identity mapping cannot be combined with rules/constants.")
        if not self.identity and not rules and not self.constants:
            raise DataIOConfigurationError("Mapping must enable identity or define rules/constants.")
        frozen_constants = deep_freeze_json(self.constants)
        if not isinstance(frozen_constants, Mapping):
            raise DataIOConfigurationError("constants must form an object.")
        _validate_target_tree(
            [rule.target_path for rule in rules], constants=self.constants, label="Object mapping"
        )
        object.__setattr__(self, "rules", rules)
        object.__setattr__(self, "constants", frozen_constants)

    @classmethod
    def identity_mapping(cls) -> "ObjectMapping":
        return cls(identity=True)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "object",
            "version": self.version,
            "identity": self.identity,
            "constants": deep_thaw_json(self.constants),
            "rules": [rule.to_dict() for rule in self.rules],
        }


@dataclass(frozen=True, slots=True)
class CsvTable:
    """One explicitly declared CSV table within a file or directory bundle."""

    target_path: str
    fields: tuple[MappingRule, ...]
    file_name: str | None = None
    delimiter: str | None = None
    encoding: str | None = None
    required_headers: tuple[str, ...] = ()
    min_rows: int = 1

    def __post_init__(self) -> None:
        target_path = _clean_target_path(self.target_path, field_name="CSV target_path", allow_empty=True)
        fields = tuple(self.fields)
        if not fields or any(not isinstance(rule, MappingRule) for rule in fields):
            raise DataIOConfigurationError("CSV fields must contain at least one MappingRule.")
        for rule in fields:
            if rule.has_constant:
                continue
            if not rule.source_path or "." in rule.source_path or "[" in rule.source_path:
                raise DataIOConfigurationError("CSV source_path must be one exact header name.")
        _validate_target_tree([rule.target_path for rule in fields], constants={}, label="CSV field mapping")
        file_name = _normalize_bundle_name(self.file_name)
        delimiter = None if self.delimiter is None else _validate_delimiter(self.delimiter, field_name="CSV delimiter")
        if self.encoding is not None and (not isinstance(self.encoding, str) or not self.encoding.strip()):
            raise DataIOConfigurationError("CSV encoding must be a non-empty string.")
        if not isinstance(self.required_headers, Sequence) or isinstance(self.required_headers, (str, bytes, bytearray)):
            raise DataIOConfigurationError("required_headers must be a sequence of strings.")
        headers = tuple(item.strip() for item in self.required_headers)
        if any(not item for item in headers) or len(headers) != len(set(headers)):
            raise DataIOConfigurationError("required_headers must be unique non-empty strings.")
        if type(self.min_rows) is not int or self.min_rows < 0:
            raise DataIOConfigurationError("min_rows must be a non-negative integer.")
        object.__setattr__(self, "target_path", target_path)
        object.__setattr__(self, "fields", fields)
        object.__setattr__(self, "file_name", file_name)
        object.__setattr__(self, "delimiter", delimiter)
        object.__setattr__(self, "encoding", None if self.encoding is None else self.encoding.strip())
        object.__setattr__(self, "required_headers", headers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "target_path": self.target_path,
            "file_name": self.file_name,
            "delimiter": self.delimiter,
            "encoding": self.encoding,
            "required_headers": list(self.required_headers),
            "min_rows": self.min_rows,
            "fields": [rule.to_dict() for rule in self.fields],
        }


@dataclass(frozen=True, slots=True)
class CsvMapping:
    """Explicit mapping for one CSV file or a declared directory bundle."""

    tables: tuple[CsvTable, ...]
    constants: Mapping[str, Any] = field(default_factory=dict)
    version: str = _MAPPING_VERSION

    def __post_init__(self) -> None:
        tables = tuple(self.tables)
        if not tables or any(not isinstance(table, CsvTable) for table in tables):
            raise DataIOConfigurationError("CSV mapping must contain at least one CsvTable.")
        if not isinstance(self.constants, Mapping):
            raise DataIOConfigurationError("CSV constants must be a mapping.")
        if self.version != _MAPPING_VERSION:
            raise DataIOConfigurationError(f"Unsupported mapping version {self.version!r}.")
        target_paths = [table.target_path for table in tables if table.target_path]
        _validate_target_tree(target_paths, constants=self.constants, label="CSV table mapping")
        names = [table.file_name.casefold() for table in tables if table.file_name is not None]
        if len(names) != len(set(names)):
            raise DataIOConfigurationError("CSV bundle file names must be unique case-insensitively.")
        frozen_constants = deep_freeze_json(self.constants)
        if not isinstance(frozen_constants, Mapping):
            raise DataIOConfigurationError("CSV constants must form an object.")
        object.__setattr__(self, "tables", tables)
        object.__setattr__(self, "constants", frozen_constants)

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": "csv",
            "version": self.version,
            "constants": deep_thaw_json(self.constants),
            "tables": [table.to_dict() for table in self.tables],
        }


@dataclass(frozen=True, slots=True)
class DataIssue:
    severity: str
    code: str
    message: str
    path: str = "source"
    row_number: int | None = None

    def __post_init__(self) -> None:
        if not all(isinstance(v, str) for v in (self.severity, self.code, self.message, self.path)):
            raise DataIOConfigurationError("Issue severity, code, message, and path must be strings.")
        severity = self.severity.strip().lower()
        if severity not in {"info", "warning"}:
            raise DataIOConfigurationError("Data issues may only be info or warning; failures raise exceptions.")
        code = self.code.strip().lower().replace(" ", "_")
        message = self.message.strip()
        path = self.path.strip()
        if not code or not message or not path:
            raise DataIOConfigurationError("Issue code, message, and path must be non-empty.")
        if self.row_number is not None and (type(self.row_number) is not int or self.row_number <= 0):
            raise DataIOConfigurationError("row_number must be a positive integer when provided.")
        object.__setattr__(self, "severity", severity)
        object.__setattr__(self, "code", code)
        object.__setattr__(self, "message", message)
        object.__setattr__(self, "path", path)

    def to_dict(self) -> dict[str, Any]:
        return {
            "severity": self.severity,
            "code": self.code,
            "message": self.message,
            "path": self.path,
            "row_number": self.row_number,
        }
