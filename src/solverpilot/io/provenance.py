"""Immutable, content-addressed provenance records for ingested data."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from pathlib import PurePosixPath
from typing import Any

from .errors import DataIOConfigurationError
from .hashing import deep_freeze_json, deep_thaw_json

_SHA256_RE = re.compile(r"^[0-9a-fA-F]{64}$")


def _sha(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or _SHA256_RE.fullmatch(value.strip()) is None:
        raise DataIOConfigurationError(f"{field_name} must be a 64-character SHA-256 digest.")
    return value.strip().lower()


def _text(value: str, *, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise DataIOConfigurationError(f"{field_name} must be a non-empty string.")
    return value.strip()


@dataclass(frozen=True, slots=True)
class SourceFileProvenance:
    name: str
    sha256: str
    size_bytes: int

    def __post_init__(self) -> None:
        if not isinstance(self.name, str):
            raise DataIOConfigurationError("source file name must be a string.")
        name = self.name.strip().replace("\\", "/")
        posix = PurePosixPath(name)
        if not name or posix.is_absolute() or any(
            part in {"", ".", ".."} or ":" in part for part in posix.parts
        ):
            raise DataIOConfigurationError("source file name must be a safe relative name.")
        if type(self.size_bytes) is not int or self.size_bytes < 0:
            raise DataIOConfigurationError("size_bytes must be a non-negative integer.")
        object.__setattr__(self, "name", posix.as_posix())
        object.__setattr__(self, "sha256", _sha(self.sha256, field_name="source file sha256"))

    def to_dict(self) -> dict[str, Any]:
        return {"name": self.name, "sha256": self.sha256, "size_bytes": self.size_bytes}


@dataclass(frozen=True, slots=True)
class SourceProvenance:
    loader: str
    loader_version: str
    source_type: str
    source_name: str
    source_sha256: str
    source_size_bytes: int
    mapping_sha256: str
    normalized_payload_sha256: str
    policy_sha256: str
    record_count: int
    encoding: str
    delimiter: str | None = None
    source_files: tuple[SourceFileProvenance, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance_schema: str = "solverpilot.source_provenance.v1"

    def __post_init__(self) -> None:
        for field_name in ("loader", "loader_version", "source_type", "source_name", "encoding", "provenance_schema"):
            object.__setattr__(self, field_name, _text(getattr(self, field_name), field_name=field_name))
        object.__setattr__(self, "source_sha256", _sha(self.source_sha256, field_name="source_sha256"))
        object.__setattr__(self, "mapping_sha256", _sha(self.mapping_sha256, field_name="mapping_sha256"))
        object.__setattr__(self, "normalized_payload_sha256", _sha(self.normalized_payload_sha256, field_name="normalized_payload_sha256"))
        object.__setattr__(self, "policy_sha256", _sha(self.policy_sha256, field_name="policy_sha256"))
        if type(self.source_size_bytes) is not int or self.source_size_bytes < 0:
            raise DataIOConfigurationError("source_size_bytes must be a non-negative integer.")
        if type(self.record_count) is not int or self.record_count < 0:
            raise DataIOConfigurationError("record_count must be a non-negative integer.")
        if self.delimiter is not None and (not isinstance(self.delimiter, str) or len(self.delimiter) != 1):
            raise DataIOConfigurationError("delimiter must be one character when provided.")
        files = tuple(self.source_files)
        if any(not isinstance(item, SourceFileProvenance) for item in files):
            raise DataIOConfigurationError("source_files must contain SourceFileProvenance values.")
        frozen = deep_freeze_json(self.metadata)
        if not isinstance(frozen, Mapping):
            raise DataIOConfigurationError("provenance metadata must form an object.")
        object.__setattr__(self, "source_files", files)
        object.__setattr__(self, "metadata", frozen)

    def to_dict(self) -> dict[str, Any]:
        return {
            "provenance_schema": self.provenance_schema,
            "loader": self.loader,
            "loader_version": self.loader_version,
            "source_type": self.source_type,
            "source_name": self.source_name,
            "source_sha256": self.source_sha256,
            "source_size_bytes": self.source_size_bytes,
            "mapping_sha256": self.mapping_sha256,
            "normalized_payload_sha256": self.normalized_payload_sha256,
            "policy_sha256": self.policy_sha256,
            "record_count": self.record_count,
            "encoding": self.encoding,
            "delimiter": self.delimiter,
            "source_files": [item.to_dict() for item in self.source_files],
            "metadata": deep_thaw_json(self.metadata),
        }

    def as_problem_metadata(self) -> dict[str, object]:
        """Return a detached metadata fragment suitable for Problem constructors."""
        return {"source_provenance": self.to_dict()}
