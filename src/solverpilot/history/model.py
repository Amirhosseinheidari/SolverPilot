from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping

from solverpilot.io.hashing import deep_freeze_json, deep_thaw_json

from .errors import HistoryDataError


def _text(value: str, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise HistoryDataError(f"{field_name} must be a non-empty string")
    return value.strip()


def _sha(value: str, field_name: str) -> str:
    text = _text(value, field_name).lower()
    if len(text) != 64 or any(ch not in "0123456789abcdef" for ch in text):
        raise HistoryDataError(f"{field_name} must be a SHA-256 digest")
    return text


def _json_object(value: Mapping[str, Any], field_name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise HistoryDataError(f"{field_name} must be a mapping")
    try:
        frozen = deep_freeze_json(value)
    except Exception as exc:  # normalize lower-layer configuration errors
        raise HistoryDataError(f"{field_name} must contain finite JSON-safe values") from exc
    if not isinstance(frozen, Mapping):
        raise HistoryDataError(f"{field_name} must form a JSON object")
    return frozen


@dataclass(frozen=True, slots=True)
class ProblemHistoryRecord:
    problem_data_hash: str
    problem_structural_hash: str
    problem_class: str
    n_variables: int
    n_constraints: int
    source_provenance: Mapping[str, Any] = field(default_factory=dict)
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "problem_data_hash", _sha(self.problem_data_hash, "problem_data_hash"))
        object.__setattr__(self, "problem_structural_hash", _sha(self.problem_structural_hash, "problem_structural_hash"))
        object.__setattr__(self, "problem_class", _text(self.problem_class, "problem_class"))
        for name in ("n_variables", "n_constraints"):
            value = getattr(self, name)
            if type(value) is not int or value < 0:
                raise HistoryDataError(f"{name} must be a non-negative integer")
        object.__setattr__(self, "source_provenance", _json_object(self.source_provenance, "source_provenance"))
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "problem_data_hash": self.problem_data_hash,
            "problem_structural_hash": self.problem_structural_hash,
            "problem_class": self.problem_class,
            "n_variables": self.n_variables,
            "n_constraints": self.n_constraints,
            "source_provenance": deep_thaw_json(self.source_provenance),
            "metadata": deep_thaw_json(self.metadata),
        }


@dataclass(frozen=True, slots=True)
class SolveHistoryRecord:
    run_id: str
    problem_data_hash: str
    problem_structural_hash: str
    problem_class: str
    backend: str | None
    backend_version: str | None
    public_status: str
    backend_status: str
    objective: float | None
    validation_valid: bool | None
    independently_verified_optimal: bool
    total_s: float | None
    environment_id: str | None = None
    trace_created_at_utc: str | None = None
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        import math

        object.__setattr__(self, "run_id", _text(self.run_id, "run_id"))
        object.__setattr__(self, "problem_data_hash", _sha(self.problem_data_hash, "problem_data_hash"))
        object.__setattr__(self, "problem_structural_hash", _sha(self.problem_structural_hash, "problem_structural_hash"))
        object.__setattr__(self, "problem_class", _text(self.problem_class, "problem_class"))
        object.__setattr__(self, "public_status", _text(self.public_status, "public_status"))
        object.__setattr__(self, "backend_status", _text(self.backend_status, "backend_status"))
        for name in ("backend", "backend_version", "environment_id", "trace_created_at_utc"):
            value = getattr(self, name)
            if value is not None:
                object.__setattr__(self, name, _text(value, name))
        if self.objective is not None and (type(self.objective) not in (int, float) or not math.isfinite(float(self.objective))):
            raise HistoryDataError("objective must be finite when provided")
        if self.total_s is not None and (type(self.total_s) not in (int, float) or not math.isfinite(float(self.total_s)) or float(self.total_s) < 0):
            raise HistoryDataError("total_s must be finite and non-negative when provided")
        if self.validation_valid is not None and type(self.validation_valid) is not bool:
            raise HistoryDataError("validation_valid must be boolean or None")
        if type(self.independently_verified_optimal) is not bool:
            raise HistoryDataError("independently_verified_optimal must be boolean")
        object.__setattr__(self, "objective", None if self.objective is None else float(self.objective))
        object.__setattr__(self, "total_s", None if self.total_s is None else float(self.total_s))
        object.__setattr__(self, "metadata", _json_object(self.metadata, "metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "problem_data_hash": self.problem_data_hash,
            "problem_structural_hash": self.problem_structural_hash,
            "problem_class": self.problem_class,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "public_status": self.public_status,
            "backend_status": self.backend_status,
            "objective": self.objective,
            "validation_valid": self.validation_valid,
            "independently_verified_optimal": self.independently_verified_optimal,
            "total_s": self.total_s,
            "environment_id": self.environment_id,
            "trace_created_at_utc": self.trace_created_at_utc,
            "metadata": deep_thaw_json(self.metadata),
        }
