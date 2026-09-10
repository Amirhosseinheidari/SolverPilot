from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
from types import MappingProxyType
from typing import Any, Mapping


class ExactnessClass(str, Enum):
    EXACT_EQUIVALENT = "exact_equivalent"
    CERTIFIED_RELAXATION = "certified_relaxation"
    HEURISTIC = "heuristic"


class MappingAvailability(str, Enum):
    IDENTITY = "identity"
    AVAILABLE = "available"
    UNAVAILABLE = "unavailable"


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return MappingProxyType({str(k): _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    return value


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]
    if isinstance(value, Enum):
        return value.value
    return value


@dataclass(frozen=True, slots=True)
class PreconditionCertificate:
    kind: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    certificate_version: str = "solverpilot.bridge-certificate.p4.v1"

    def __post_init__(self) -> None:
        object.__setattr__(self, "payload", _freeze(self.payload))

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            {"version": self.certificate_version, "kind": self.kind, "payload": _jsonable(self.payload)},
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode()
        return hashlib.sha256(raw).hexdigest()

    def to_dict(self) -> dict[str, Any]:
        return {
            "kind": self.kind,
            "payload": _jsonable(self.payload),
            "certificate_version": self.certificate_version,
            "sha256": self.sha256,
        }


@dataclass(frozen=True, slots=True)
class TransformationStep:
    step_id: str
    bridge_id: str
    bridge_version: str
    source_entity_ids: tuple[str, ...]
    generated_target_ids: tuple[str, ...]
    exactness: ExactnessClass
    preconditions: tuple[PreconditionCertificate, ...] = ()
    primal_mapping: MappingAvailability = MappingAvailability.IDENTITY
    dual_mapping: MappingAvailability = MappingAvailability.UNAVAILABLE
    certificate_mapping: MappingAvailability = MappingAvailability.UNAVAILABLE
    mutation_compatibility: tuple[str, ...] = ()
    size_delta: Mapping[str, int] = field(default_factory=dict)
    numerical_risk: Mapping[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "size_delta", _freeze(self.size_delta))
        object.__setattr__(self, "numerical_risk", _freeze(self.numerical_risk))

    def to_dict(self) -> dict[str, Any]:
        return {
            "step_id": self.step_id,
            "bridge_id": self.bridge_id,
            "bridge_version": self.bridge_version,
            "source_entity_ids": list(self.source_entity_ids),
            "generated_target_ids": list(self.generated_target_ids),
            "exactness": self.exactness.value,
            "preconditions": [c.to_dict() for c in self.preconditions],
            "primal_mapping": self.primal_mapping.value,
            "dual_mapping": self.dual_mapping.value,
            "certificate_mapping": self.certificate_mapping.value,
            "mutation_compatibility": list(self.mutation_compatibility),
            "size_delta": dict(self.size_delta),
            "numerical_risk": _jsonable(self.numerical_risk),
            "notes": list(self.notes),
        }


@dataclass(frozen=True, slots=True)
class TransformationTape:
    steps: tuple[TransformationStep, ...] = ()
    schema_version: str = "solverpilot.transformation-tape.p4.v1"

    def to_dicts(self) -> tuple[dict[str, Any], ...]:
        return tuple(step.to_dict() for step in self.steps)

    @property
    def certificate_hashes(self) -> tuple[str, ...]:
        return tuple(cert.sha256 for step in self.steps for cert in step.preconditions)

    @property
    def plan_sensitive_dependencies(self) -> tuple[str, ...]:
        deps: list[str] = []
        for step in self.steps:
            for cert in step.preconditions:
                raw = cert.payload.get("parameter_dependencies", ())
                for dep in raw:
                    dep = str(dep)
                    if dep not in deps:
                        deps.append(dep)
        return tuple(deps)


@dataclass(frozen=True, slots=True)
class BridgePolicy:
    mode: str = "safe"
    allow_certified_relaxations: bool = False
    allow_heuristic: bool = False
    require_finite_certified_big_m: bool = True
    policy_version: str = "solverpilot.bridge-policy.p4.safe.v1"

    def permits(self, exactness: ExactnessClass) -> bool:
        if exactness is ExactnessClass.EXACT_EQUIVALENT:
            return True
        if exactness is ExactnessClass.CERTIFIED_RELAXATION:
            return self.allow_certified_relaxations
        return self.allow_heuristic
