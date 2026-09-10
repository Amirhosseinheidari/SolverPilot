"""Collision-safe explicit field mapping and scalar normalization."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from typing import Any

from .errors import DataIOConfigurationError, DataIOMappingError
from .hashing import canonicalize_json_value, deep_thaw_json
from .model import MappingRule, ObjectMapping
from .numeric import parse_boolean, parse_integer, parse_number
from .strict_json import loads_strict_json

_MISSING = object()
_PATH_TOKEN_RE = re.compile(r"([^.[\]]+)|\[([0-9]+)\]")


def _path_tokens(path: str) -> tuple[str | int, ...]:
    if path == "":
        return ()
    tokens: list[str | int] = []
    position = 0
    while position < len(path):
        match = _PATH_TOKEN_RE.match(path, position)
        if match is None:
            raise DataIOMappingError(f"Invalid source path syntax at position {position}.")
        token = match.group(1)
        tokens.append(token if token is not None else int(match.group(2)))
        position = match.end()
        if position == len(path):
            break
        if path[position] == "[":
            continue
        if path[position] != ".":
            raise DataIOMappingError(f"Invalid source path syntax at position {position}.")
        position += 1
        if position == len(path) or path[position] in ".[":
            raise DataIOMappingError(f"Invalid source path syntax at position {position}.")
    return tuple(tokens)


def get_path(source: Any, path: str) -> Any:
    current = source
    for token in _path_tokens(path):
        if isinstance(token, int):
            if not isinstance(current, Sequence) or isinstance(current, (str, bytes, bytearray)):
                return _MISSING
            if token >= len(current):
                return _MISSING
            current = current[token]
        else:
            if not isinstance(current, Mapping) or token not in current:
                return _MISSING
            current = current[token]
    return current


def set_path(target: dict[str, Any], path: str, value: Any) -> None:
    if not path:
        raise DataIOMappingError("Target path cannot be empty for object field mapping.")
    parts = path.split(".")
    if any(not part or "[" in part or "]" in part for part in parts):
        raise DataIOMappingError(f"Invalid target path syntax: {path!r}.")
    current = target
    for part in parts[:-1]:
        if part not in current:
            nested: dict[str, Any] = {}
            current[part] = nested
            current = nested
            continue
        existing = current[part]
        if not isinstance(existing, dict):
            raise DataIOMappingError(
                f"Target path {path!r} collides with a non-object value at {part!r}."
            )
        current = existing
    leaf = parts[-1]
    if leaf in current:
        raise DataIOMappingError(f"Target path {path!r} would overwrite an existing value.")
    current[leaf] = value


def _canonical_mapped(value: Any, *, field_name: str) -> Any:
    try:
        return canonicalize_json_value(value)
    except DataIOConfigurationError as exc:
        raise DataIOMappingError(f"{field_name} is not a supported JSON-compatible value.") from exc


def normalize_value(
    value: Any,
    rule: MappingRule,
    *,
    field_name: str,
    strict_numbers: bool,
    max_json_depth: int,
) -> Any:
    is_null = value is None or (
        isinstance(value, str) and value.strip().lower() in rule.null_values
    )
    if is_null:
        if rule.has_default:
            value = deep_thaw_json(rule.default)
        elif rule.required:
            raise DataIOMappingError(f"{field_name} is required and cannot be null.")
        else:
            return None
    if rule.value_type == "any":
        if isinstance(value, str) and rule.trim:
            return value.strip()
        return _canonical_mapped(value, field_name=field_name)
    if rule.value_type == "string":
        if not isinstance(value, str):
            raise DataIOMappingError(f"{field_name} must be a string.")
        return value.strip() if rule.trim else value
    if rule.value_type == "integer":
        return parse_integer(value, field_name=field_name)
    if rule.value_type == "number":
        return parse_number(value, field_name=field_name, strict_binary64=strict_numbers)
    if rule.value_type == "boolean":
        return parse_boolean(value, field_name=field_name)
    if rule.value_type == "json":
        if isinstance(value, str):
            value = loads_strict_json(
                value,
                context=f"{field_name} JSON text",
                max_depth=max_json_depth,
                strict_numbers=strict_numbers,
            )
        return _canonical_mapped(value, field_name=field_name)
    raise DataIOMappingError(f"Unsupported normalization type {rule.value_type!r}.")


def apply_rules(
    source: Any,
    rules: Sequence[MappingRule],
    *,
    strict_numbers: bool,
    max_json_depth: int,
    row_number: int | None = None,
) -> dict[str, Any]:
    target: dict[str, Any] = {}
    for rule in rules:
        if rule.has_constant:
            raw_value = deep_thaw_json(rule.constant)
        else:
            if rule.source_path is None:
                raise DataIOMappingError("Mapping rule without a constant requires source_path.")
            raw_value = get_path(source, rule.source_path)
            if raw_value is _MISSING:
                if rule.has_default:
                    raw_value = deep_thaw_json(rule.default)
                elif rule.required:
                    suffix = "" if row_number is None else f" at row {row_number}"
                    raise DataIOMappingError(
                        f"Required source field {rule.source_path!r} is missing{suffix}."
                    )
                else:
                    raw_value = None
        suffix = "" if row_number is None else f" at row {row_number}"
        normalized = normalize_value(
            raw_value,
            rule,
            field_name=f"Field {rule.source_path or rule.target_path!r}{suffix}",
            strict_numbers=strict_numbers,
            max_json_depth=max_json_depth,
        )
        set_path(target, rule.target_path, normalized)
    return target


def apply_object_mapping(
    source: Any,
    config: ObjectMapping,
    *,
    strict_numbers: bool,
    max_json_depth: int,
) -> Any:
    if config.identity:
        return _canonical_mapped(source, field_name="Identity-mapped source")
    target = _canonical_mapped(deep_thaw_json(config.constants), field_name="Object mapping constants")
    if not isinstance(target, dict):
        raise DataIOMappingError("Object mapping constants must form an object.")
    mapped = apply_rules(
        source,
        config.rules,
        strict_numbers=strict_numbers,
        max_json_depth=max_json_depth,
    )
    for key, value in mapped.items():
        if key in target:
            raise DataIOMappingError(f"Object mapping would overwrite constant key {key!r}.")
        target[key] = value
    return target
