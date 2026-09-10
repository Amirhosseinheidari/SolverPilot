"""Canonical JSON snapshots and SHA-256 helpers for data provenance."""

from __future__ import annotations

import hashlib
import json
import math
from collections.abc import Mapping, Sequence
from pathlib import Path
from types import MappingProxyType
from typing import Any

from .errors import DataIOConfigurationError


def _validate_text(value: str) -> str:
    try:
        value.encode("utf-8")
    except UnicodeEncodeError as exc:
        raise DataIOConfigurationError("JSON-compatible text contains an invalid Unicode surrogate.") from exc
    return value


def canonicalize_json_value(value: Any) -> Any:
    """Return a detached JSON-compatible value with deterministic mapping order."""
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return _validate_text(value)
    if isinstance(value, int):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataIOConfigurationError("JSON-compatible numbers must be finite.")
        return value
    if isinstance(value, Mapping):
        keys = tuple(value.keys())
        if any(not isinstance(key, str) for key in keys):
            raise DataIOConfigurationError("Canonical JSON mapping keys must be strings.")
        return {
            _validate_text(key): canonicalize_json_value(value[key])
            for key in sorted(keys)
        }
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [canonicalize_json_value(item) for item in value]
    raise DataIOConfigurationError(
        f"Value of type {type(value).__name__} is not JSON-compatible for provenance."
    )


def _freeze(value: Any) -> Any:
    if isinstance(value, dict):
        return MappingProxyType({key: _freeze(item) for key, item in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(item) for item in value)
    return value


def deep_freeze_json(value: Any) -> Any:
    """Create a detached, recursively immutable JSON-compatible snapshot."""
    try:
        return _freeze(canonicalize_json_value(value))
    except DataIOConfigurationError:
        raise
    except RecursionError as exc:
        raise DataIOConfigurationError(
            "JSON-compatible configuration exceeds supported nesting or contains a cycle."
        ) from exc


def deep_thaw_json(value: Any) -> Any:
    """Return a detached mutable JSON-compatible copy of a frozen snapshot."""
    if isinstance(value, Mapping):
        return {str(key): deep_thaw_json(item) for key, item in value.items()}
    if isinstance(value, Sequence) and not isinstance(value, (str, bytes, bytearray)):
        return [deep_thaw_json(item) for item in value]
    return canonicalize_json_value(value)


def canonical_json_bytes(value: Any) -> bytes:
    try:
        normalized = canonicalize_json_value(value)
        return json.dumps(
            normalized,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    except DataIOConfigurationError:
        raise
    except (TypeError, ValueError, UnicodeEncodeError, RecursionError) as exc:
        raise DataIOConfigurationError("Value cannot be serialized as canonical JSON.") from exc


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(value: Any) -> str:
    return sha256_bytes(canonical_json_bytes(value))


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()
