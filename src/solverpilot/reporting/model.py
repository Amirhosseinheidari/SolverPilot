from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Any

from solverpilot.io.hashing import deep_freeze_json, deep_thaw_json


REPORT_SCHEMA_VERSION = "0.1"


class ClaimKind(str, Enum):
    FEASIBILITY = "feasibility"
    OPTIMALITY = "optimality"
    INFEASIBILITY = "infeasibility"
    UNBOUNDEDNESS = "unboundedness"
    PLANNER_SELECTION = "planner_selection"
    REOPTIMIZATION = "reoptimization"


class ClaimDisposition(str, Enum):
    SUPPORTED = "supported"
    QUALIFIED = "qualified"
    BLOCKED = "blocked"


def _nonempty_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field_name} must be a non-empty string")
    return value.strip()


def _tuple_text(values: Sequence[str], field_name: str) -> tuple[str, ...]:
    if isinstance(values, (str, bytes, bytearray)):
        raise ValueError(f"{field_name} must be a sequence of strings, not a string")
    try:
        normalized = tuple(_nonempty_text(value, field_name) for value in values)
    except TypeError as exc:
        raise ValueError(f"{field_name} must be a sequence of strings") from exc
    return normalized


def _freeze_mapping(value: Mapping[str, Any] | None, field_name: str) -> Mapping[str, Any] | None:
    if value is None:
        return None
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a mapping or None")
    try:
        frozen = deep_freeze_json(value)
    except Exception as exc:
        if "finite" in str(exc).lower():
            raise ValueError(f"report contains NaN or infinity at $.{field_name}") from exc
        raise ValueError(f"report is not JSON-safe at $.{field_name}") from exc
    if not isinstance(frozen, Mapping):
        raise ValueError(f"{field_name} must form an object")
    return frozen


@dataclass(frozen=True, slots=True)
class ExplanationClaim:
    claim_id: str
    kind: ClaimKind
    disposition: ClaimDisposition
    statement: str
    rationale: str
    evidence_refs: tuple[str, ...] = ()
    qualifiers: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        claim_id = _nonempty_text(self.claim_id, "claim_id")
        statement = _nonempty_text(self.statement, "statement")
        rationale = _nonempty_text(self.rationale, "rationale")
        try:
            kind = self.kind if isinstance(self.kind, ClaimKind) else ClaimKind(str(self.kind).strip().lower())
        except ValueError as exc:
            raise ValueError("kind must be a supported ClaimKind") from exc
        try:
            disposition = (
                self.disposition
                if isinstance(self.disposition, ClaimDisposition)
                else ClaimDisposition(str(self.disposition).strip().lower())
            )
        except ValueError as exc:
            raise ValueError("disposition must be a supported ClaimDisposition") from exc
        evidence_refs = _tuple_text(self.evidence_refs, "evidence_refs")
        qualifiers = _tuple_text(self.qualifiers, "qualifiers")
        if len(set(evidence_refs)) != len(evidence_refs):
            raise ValueError("evidence_refs must not contain duplicates")
        object.__setattr__(self, "claim_id", claim_id)
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "disposition", disposition)
        object.__setattr__(self, "statement", statement)
        object.__setattr__(self, "rationale", rationale)
        object.__setattr__(self, "evidence_refs", evidence_refs)
        object.__setattr__(self, "qualifiers", qualifiers)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim_id": self.claim_id,
            "kind": self.kind.value,
            "disposition": self.disposition.value,
            "statement": self.statement,
            "rationale": self.rationale,
            "evidence_refs": list(self.evidence_refs),
            "qualifiers": list(self.qualifiers),
        }


@dataclass(frozen=True, slots=True)
class ExplanationReport:
    status: str
    summary: str
    status_explanation: str
    problem_class: str
    backend: str | None
    backend_version: str | None
    objective: float | None
    validation: Mapping[str, Any] | None
    optimality: Mapping[str, Any]
    planner: Mapping[str, Any] | None
    runtime: Mapping[str, Any]
    diagnostics: Mapping[str, Any] | None
    claims: tuple[ExplanationClaim, ...]
    warnings: tuple[str, ...] = ()
    schema_version: str = REPORT_SCHEMA_VERSION

    def __post_init__(self) -> None:
        status = _nonempty_text(self.status, "status")
        summary = _nonempty_text(self.summary, "summary")
        status_explanation = _nonempty_text(self.status_explanation, "status_explanation")
        problem_class = _nonempty_text(self.problem_class, "problem_class")
        backend = None if self.backend is None else _nonempty_text(self.backend, "backend")
        backend_version = None if self.backend_version is None else _nonempty_text(self.backend_version, "backend_version")
        if self.schema_version != REPORT_SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {REPORT_SCHEMA_VERSION!r}")
        if self.objective is not None:
            import math

            if isinstance(self.objective, bool) or type(self.objective) not in (int, float) or not math.isfinite(float(self.objective)):
                raise ValueError("report contains NaN or infinity at $.objective")
            objective: float | None = float(self.objective)
        else:
            objective = None
        validation = _freeze_mapping(self.validation, "validation")
        optimality = _freeze_mapping(self.optimality, "optimality")
        planner = _freeze_mapping(self.planner, "planner")
        runtime = _freeze_mapping(self.runtime, "runtime")
        diagnostics = _freeze_mapping(self.diagnostics, "diagnostics")
        if optimality is None or runtime is None:
            raise ValueError("optimality and runtime must be mappings")
        claims = tuple(self.claims)
        if any(not isinstance(claim, ExplanationClaim) for claim in claims):
            raise ValueError("claims must contain ExplanationClaim objects")
        warnings = _tuple_text(self.warnings, "warnings")
        ids = [claim.claim_id for claim in claims]
        if len(ids) != len(set(ids)):
            raise ValueError("claim IDs must be unique")
        object.__setattr__(self, "status", status)
        object.__setattr__(self, "summary", summary)
        object.__setattr__(self, "status_explanation", status_explanation)
        object.__setattr__(self, "problem_class", problem_class)
        object.__setattr__(self, "backend", backend)
        object.__setattr__(self, "backend_version", backend_version)
        object.__setattr__(self, "objective", objective)
        object.__setattr__(self, "validation", validation)
        object.__setattr__(self, "optimality", optimality)
        object.__setattr__(self, "planner", planner)
        object.__setattr__(self, "runtime", runtime)
        object.__setattr__(self, "diagnostics", diagnostics)
        object.__setattr__(self, "claims", claims)
        object.__setattr__(self, "warnings", warnings)
        _validate_claim_evidence_refs(claims)
        if not self.claim_safe:
            raise ValueError("ExplanationReport contains a supported claim that is not authorized by structured evidence")
        _assert_json_safe(self.to_dict())

    @property
    def claim_safe(self) -> bool:
        for claim in self.claims:
            if claim.disposition is not ClaimDisposition.SUPPORTED:
                continue
            if claim.kind is ClaimKind.FEASIBILITY:
                if not (isinstance(self.validation, Mapping) and self.validation.get("valid") is True):
                    return False
            elif claim.kind is ClaimKind.OPTIMALITY:
                if self.optimality.get("independently_verified_optimal") is not True:
                    return False
            elif claim.kind is ClaimKind.INFEASIBILITY:
                if not (isinstance(self.diagnostics, Mapping) and self.diagnostics.get("confirmed_infeasible") is True):
                    return False
            elif claim.kind is ClaimKind.UNBOUNDEDNESS:
                # S5/S10 has no generic independent unboundedness-proof field.
                return False
            elif claim.kind is ClaimKind.PLANNER_SELECTION:
                if self.planner is None:
                    return False
            elif claim.kind is ClaimKind.REOPTIMIZATION:
                if self.runtime.get("reuse_applied") is not True:
                    return False
        return True

    def to_dict(self) -> dict[str, Any]:
        payload = {
            "status": self.status,
            "summary": self.summary,
            "status_explanation": self.status_explanation,
            "problem_class": self.problem_class,
            "backend": self.backend,
            "backend_version": self.backend_version,
            "objective": self.objective,
            "validation": None if self.validation is None else deep_thaw_json(self.validation),
            "optimality": deep_thaw_json(self.optimality),
            "planner": None if self.planner is None else deep_thaw_json(self.planner),
            "runtime": deep_thaw_json(self.runtime),
            "diagnostics": None if self.diagnostics is None else deep_thaw_json(self.diagnostics),
            "claims": [claim.to_dict() for claim in self.claims],
            "warnings": list(self.warnings),
            "schema_version": self.schema_version,
        }
        _assert_json_safe(payload)
        if not self.claim_safe:
            raise ValueError("ExplanationReport no longer satisfies claim-safety invariants")
        return payload


def _assert_json_safe(value: Any, *, path: str = "$") -> None:
    import math

    if value is None or isinstance(value, (str, bool, int)):
        return
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError(f"report contains NaN or infinity at {path}")
        return
    if isinstance(value, (tuple, list)):
        for index, item in enumerate(value):
            _assert_json_safe(item, path=f"{path}[{index}]")
        return
    if isinstance(value, Mapping):
        for key, item in value.items():
            if not isinstance(key, str):
                raise ValueError(f"report mapping keys must be strings at {path}")
            _assert_json_safe(item, path=f"{path}.{key}")
        return
    raise ValueError(f"report is not JSON-safe at {path}: {type(value).__name__}")


_KNOWN_EVIDENCE_REFS = frozenset({
    "result.status",
    "result.validation",
    "result.optimality_evidence",
    "result.backend_status",
    "result.diagnostics",
    "result.plan",
    "result.trace",
})


def _validate_claim_evidence_refs(claims: tuple[ExplanationClaim, ...]) -> None:
    unknown = sorted({ref for claim in claims for ref in claim.evidence_refs if ref not in _KNOWN_EVIDENCE_REFS})
    if unknown:
        raise ValueError(f"claims contain unknown evidence references: {unknown}")
