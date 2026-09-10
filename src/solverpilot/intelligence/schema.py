"""Versioned, immutable, content-addressed feature schemas and records."""
from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import Enum
import math
import re
from types import MappingProxyType
from typing import Any

from solverpilot.io.hashing import sha256_json

from .errors import FeatureSchemaError

_SEMVER = re.compile(r"^(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)\.(0|[1-9][0-9]*)(?:[-+][0-9A-Za-z.-]+)?$")
_SHA256 = re.compile(r"^[0-9a-f]{64}$")


class FeatureDataType(str, Enum):
    INTEGER = "integer"
    NUMBER = "number"
    BOOLEAN = "boolean"
    CATEGORICAL = "categorical"


class MissingnessPolicy(str, Enum):
    FORBIDDEN = "forbidden"
    NULLABLE = "nullable"
    NOT_APPLICABLE = "not_applicable"


def _text(value: Any, *, field_name: str, lower: bool = False) -> str:
    text = str(value).strip()
    if not text:
        raise FeatureSchemaError(f"{field_name} must be non-empty")
    return text.lower() if lower else text


def _semver(value: str) -> str:
    text = _text(value, field_name="schema_version")
    if _SEMVER.fullmatch(text) is None:
        raise FeatureSchemaError("feature schema version must be semantic version text")
    return text


def _finite(value: Any, *, field_name: str) -> float:
    if type(value) not in (int, float) or not math.isfinite(float(value)):
        raise FeatureSchemaError(f"{field_name} must be finite")
    return float(value)


def _sha256(value: str, *, field_name: str) -> str:
    text = _text(value, field_name=field_name, lower=True)
    if _SHA256.fullmatch(text) is None:
        raise FeatureSchemaError(f"{field_name} must be a lowercase SHA-256 digest")
    return text


@dataclass(frozen=True, slots=True)
class FeatureDefinition:
    name: str
    data_type: FeatureDataType | str
    unit: str
    missingness: MissingnessPolicy | str
    description: str
    source_fields: tuple[str, ...] | Sequence[str]
    lower_bound: float | None = None
    upper_bound: float | None = None
    allowed_values: tuple[str, ...] | Sequence[str] = ()

    def __post_init__(self) -> None:
        name = _text(self.name, field_name="feature.name", lower=True)
        try:
            dtype = self.data_type if isinstance(self.data_type, FeatureDataType) else FeatureDataType(str(self.data_type).strip().lower())
        except ValueError as exc:
            raise FeatureSchemaError(f"unsupported feature data type for {name!r}") from exc
        try:
            missingness = self.missingness if isinstance(self.missingness, MissingnessPolicy) else MissingnessPolicy(str(self.missingness).strip().lower())
        except ValueError as exc:
            raise FeatureSchemaError(f"unsupported missingness policy for {name!r}") from exc
        unit = _text(self.unit, field_name=f"feature {name} unit", lower=True)
        description = _text(self.description, field_name=f"feature {name} description")
        fields = tuple(sorted(_text(x, field_name=f"feature {name} source field") for x in self.source_fields))
        if not fields or len(fields) != len(set(fields)):
            raise FeatureSchemaError(f"feature {name} source_fields must be non-empty and unique")
        lower = None if self.lower_bound is None else _finite(self.lower_bound, field_name=f"feature {name} lower_bound")
        upper = None if self.upper_bound is None else _finite(self.upper_bound, field_name=f"feature {name} upper_bound")
        if lower is not None and upper is not None and lower > upper:
            raise FeatureSchemaError(f"feature {name} bounds are inverted")
        allowed = tuple(sorted(_text(x, field_name=f"feature {name} allowed value") for x in self.allowed_values))
        if len(allowed) != len(set(allowed)):
            raise FeatureSchemaError(f"feature {name} allowed_values contain duplicates")
        if dtype is FeatureDataType.CATEGORICAL and not allowed:
            raise FeatureSchemaError(f"categorical feature {name} requires allowed_values")
        if dtype is not FeatureDataType.CATEGORICAL and allowed:
            raise FeatureSchemaError(f"non-categorical feature {name} cannot declare allowed_values")
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "data_type", dtype)
        object.__setattr__(self, "unit", unit)
        object.__setattr__(self, "missingness", missingness)
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "source_fields", fields)
        object.__setattr__(self, "lower_bound", lower)
        object.__setattr__(self, "upper_bound", upper)
        object.__setattr__(self, "allowed_values", allowed)

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "data_type": self.data_type.value,
            "unit": self.unit,
            "missingness": self.missingness.value,
            "description": self.description,
            "source_fields": list(self.source_fields),
            "lower_bound": self.lower_bound,
            "upper_bound": self.upper_bound,
            "allowed_values": list(self.allowed_values),
        }


@dataclass(frozen=True, slots=True)
class FeatureSchema:
    domain: str
    version: str
    definitions: tuple[FeatureDefinition, ...] | Sequence[FeatureDefinition]
    feature_schema_id: str = field(init=False)

    def __post_init__(self) -> None:
        domain = _text(self.domain, field_name="domain", lower=True)
        version = _semver(self.version)
        definitions = tuple(sorted(self.definitions, key=lambda x: x.name))
        if not definitions or any(not isinstance(x, FeatureDefinition) for x in definitions):
            raise FeatureSchemaError("definitions must contain at least one FeatureDefinition")
        names = tuple(x.name for x in definitions)
        if len(names) != len(set(names)):
            raise FeatureSchemaError("feature names must be unique")
        payload = {"domain": domain, "version": version, "definitions": [x.to_dict() for x in definitions]}
        object.__setattr__(self, "domain", domain)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "definitions", definitions)
        object.__setattr__(self, "feature_schema_id", sha256_json(payload))

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(x.name for x in self.definitions)

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain": self.domain,
            "version": self.version,
            "feature_schema_id": self.feature_schema_id,
            "definitions": [x.to_dict() for x in self.definitions],
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> "FeatureSchema":
        required = {"domain", "version", "feature_schema_id", "definitions"}
        if set(payload) != required:
            raise FeatureSchemaError("serialized FeatureSchema keys do not match contract")
        definitions = tuple(FeatureDefinition(**dict(x)) for x in payload["definitions"])
        rebuilt = cls(str(payload["domain"]), str(payload["version"]), definitions)
        if rebuilt.feature_schema_id != str(payload["feature_schema_id"]):
            raise FeatureSchemaError("serialized feature_schema_id does not match canonical content")
        return rebuilt


@dataclass(frozen=True, slots=True)
class FeatureRecord:
    instance_id: str
    feature_schema_id: str
    feature_schema_version: str
    values: Mapping[str, Any]
    missingness_reasons: Mapping[str, str]
    source_content_sha256: str
    extractor_id: str
    extractor_version: str
    record_id: str = field(init=False)

    def __post_init__(self) -> None:
        instance_id = _text(self.instance_id, field_name="instance_id")
        schema_id = _sha256(self.feature_schema_id, field_name="feature_schema_id")
        schema_version = _semver(self.feature_schema_version)
        source_hash = _sha256(self.source_content_sha256, field_name="source_content_sha256")
        extractor_id = _text(self.extractor_id, field_name="extractor_id", lower=True)
        extractor_version = _semver(self.extractor_version)
        values = MappingProxyType(dict(self.values))
        reasons = MappingProxyType({str(k): _text(v, field_name=f"missingness reason {k}") for k, v in self.missingness_reasons.items()})
        payload = {
            "instance_id": instance_id,
            "feature_schema_id": schema_id,
            "feature_schema_version": schema_version,
            "values": dict(values),
            "missingness_reasons": dict(reasons),
            "source_content_sha256": source_hash,
            "extractor_id": extractor_id,
            "extractor_version": extractor_version,
        }
        object.__setattr__(self, "instance_id", instance_id)
        object.__setattr__(self, "feature_schema_id", schema_id)
        object.__setattr__(self, "feature_schema_version", schema_version)
        object.__setattr__(self, "values", values)
        object.__setattr__(self, "missingness_reasons", reasons)
        object.__setattr__(self, "source_content_sha256", source_hash)
        object.__setattr__(self, "extractor_id", extractor_id)
        object.__setattr__(self, "extractor_version", extractor_version)
        object.__setattr__(self, "record_id", sha256_json(payload))

    def to_dict(self) -> dict[str, Any]:
        return {
            "record_id": self.record_id,
            "instance_id": self.instance_id,
            "feature_schema_id": self.feature_schema_id,
            "feature_schema_version": self.feature_schema_version,
            "values": dict(self.values),
            "missingness_reasons": dict(self.missingness_reasons),
            "source_content_sha256": self.source_content_sha256,
            "extractor_id": self.extractor_id,
            "extractor_version": self.extractor_version,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any], *, schema: FeatureSchema) -> "FeatureRecord":
        required = {
            "record_id", "instance_id", "feature_schema_id", "feature_schema_version", "values",
            "missingness_reasons", "source_content_sha256", "extractor_id", "extractor_version",
        }
        if set(payload) != required:
            raise FeatureSchemaError("serialized FeatureRecord keys do not match contract")
        rebuilt = build_feature_record(
            schema=schema,
            instance_id=str(payload["instance_id"]),
            values=dict(payload["values"]),
            missingness_reasons=dict(payload["missingness_reasons"]),
            source_content_sha256=str(payload["source_content_sha256"]),
            extractor_id=str(payload["extractor_id"]),
            extractor_version=str(payload["extractor_version"]),
        )
        if rebuilt.record_id != str(payload["record_id"]):
            raise FeatureSchemaError("serialized record_id does not match canonical content")
        return rebuilt


def build_feature_schema(*, domain: str, version: str, definitions: Sequence[FeatureDefinition]) -> FeatureSchema:
    return FeatureSchema(domain=domain, version=version, definitions=tuple(definitions))


def build_feature_record(
    *,
    schema: FeatureSchema,
    instance_id: str,
    values: Mapping[str, Any],
    source_content_sha256: str,
    extractor_id: str,
    extractor_version: str,
    missingness_reasons: Mapping[str, str] | None = None,
) -> FeatureRecord:
    definitions = {x.name: x for x in schema.definitions}
    if set(values) != set(definitions):
        raise FeatureSchemaError(
            f"feature values must exactly match schema names; missing={sorted(set(definitions)-set(values))}, extra={sorted(set(values)-set(definitions))}"
        )
    reasons = dict(missingness_reasons or {})
    normalized: dict[str, Any] = {}
    for name, definition in definitions.items():
        value = values[name]
        if value is None:
            if definition.missingness is MissingnessPolicy.FORBIDDEN:
                raise FeatureSchemaError(f"feature {name} may not be missing")
            if name not in reasons:
                raise FeatureSchemaError(f"missing feature {name} requires a reason")
            normalized[name] = None
            continue
        if name in reasons:
            raise FeatureSchemaError(f"present feature {name} may not have a missingness reason")
        normalized[name] = _validate_value(value, definition)
    missing = {k for k, v in normalized.items() if v is None}
    if set(reasons) - missing:
        raise FeatureSchemaError("missingness_reasons contains non-missing feature names")
    return FeatureRecord(
        instance_id=instance_id,
        feature_schema_id=schema.feature_schema_id,
        feature_schema_version=schema.version,
        values=normalized,
        missingness_reasons=reasons,
        source_content_sha256=source_content_sha256,
        extractor_id=extractor_id,
        extractor_version=extractor_version,
    )


def _validate_value(value: Any, definition: FeatureDefinition) -> Any:
    if definition.data_type is FeatureDataType.BOOLEAN:
        if type(value) is not bool:
            raise FeatureSchemaError(f"{definition.name} must be boolean")
        return value
    if definition.data_type is FeatureDataType.INTEGER:
        if type(value) is not int:
            raise FeatureSchemaError(f"{definition.name} must be integer")
        number = float(value)
    elif definition.data_type is FeatureDataType.NUMBER:
        if type(value) not in (int, float) or not math.isfinite(float(value)):
            raise FeatureSchemaError(f"{definition.name} must be finite numeric")
        number = float(value)
    else:
        text = _text(value, field_name=definition.name)
        if text not in definition.allowed_values:
            raise FeatureSchemaError(f"{definition.name} has unsupported categorical value {text!r}")
        return text
    if definition.lower_bound is not None and number < definition.lower_bound:
        raise FeatureSchemaError(f"{definition.name} is below lower bound")
    if definition.upper_bound is not None and number > definition.upper_bound:
        raise FeatureSchemaError(f"{definition.name} is above upper bound")
    return int(value) if definition.data_type is FeatureDataType.INTEGER else number
