from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from math import isclose, isfinite
from typing import Iterable, Mapping, Any


_REFERENCE_SUCCESS = {"objective_matches_optimum"}


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"

    @classmethod
    def parse(cls, value: str | "ObjectiveSense") -> "ObjectiveSense":
        if isinstance(value, cls):
            return value
        normalized = str(value).strip().lower()
        aliases = {"min": cls.MINIMIZE, "max": cls.MAXIMIZE}
        if normalized in aliases:
            return aliases[normalized]
        try:
            return cls(normalized)
        except ValueError as exc:
            raise ValueError("objective_sense must be 'minimize' or 'maximize'") from exc


class OracleStatus(str, Enum):
    AVAILABLE = "available"
    NO_ELIGIBLE_CANDIDATE = "no_eligible_candidate"
    PROTOCOL_MISMATCH = "protocol_mismatch"
    ENVIRONMENT_MISMATCH = "environment_mismatch"
    INSTANCE_IDENTITY_MISMATCH = "instance_identity_mismatch"


class CandidateExclusionReason(str, Enum):
    AUTO_BACKEND = "auto_backend"
    NOT_SOLVED = "not_solved"
    NOT_VALIDATED = "not_validated"
    NOT_EXACT_SUCCESS = "not_exact_success"
    MISSING_OBJECTIVE = "missing_objective"
    NONFINITE_OBJECTIVE = "nonfinite_objective"
    MISSING_RUNTIME = "missing_runtime"
    NONFINITE_RUNTIME = "nonfinite_runtime"
    NEGATIVE_RUNTIME = "negative_runtime"


@dataclass(frozen=True, slots=True)
class OraclePolicy:
    objective_sense: ObjectiveSense | str = ObjectiveSense.MINIMIZE
    objective_atol: float = 1e-7
    objective_rtol: float = 1e-7
    runtime_tie_breaker: bool = False
    runtime_tolerance_s: float = 1e-9
    require_same_protocol: bool = True
    require_same_environment: bool = True
    require_same_instance_sha256: bool = True
    include_auto_backend: bool = False
    runtime_field: str = "wall_s"

    def __post_init__(self) -> None:
        object.__setattr__(self, "objective_sense", ObjectiveSense.parse(self.objective_sense))
        for name in ("objective_atol", "objective_rtol", "runtime_tolerance_s"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or type(raw) not in (int, float):
                raise ValueError(f"{name} must be numeric")
            value = float(raw)
            if not isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, value)
        for name in (
            "runtime_tie_breaker",
            "require_same_protocol",
            "require_same_environment",
            "require_same_instance_sha256",
            "include_auto_backend",
        ):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f"{name} must be boolean")
        if not isinstance(self.runtime_field, str) or not self.runtime_field.strip():
            raise ValueError("runtime_field must be a non-empty dotted field name")
        object.__setattr__(self, "runtime_field", self.runtime_field.strip())


@dataclass(frozen=True, slots=True)
class CandidateExclusion:
    run_id: str
    backend: str
    reason: CandidateExclusionReason
    detail: str


@dataclass(frozen=True, slots=True)
class OracleCandidate:
    run_id: str
    backend: str
    objective: float
    runtime_s: float


@dataclass(frozen=True, slots=True)
class OracleEntry:
    instance: str
    instance_sha256: str | None
    protocol_id: str | None
    environment_id: str | None
    status: OracleStatus
    objective_sense: ObjectiveSense
    oracle_objective: float | None
    oracle_run_ids: tuple[str, ...]
    eligible_run_ids: tuple[str, ...]
    fastest_oracle_runtime_s: float | None
    candidates: tuple[OracleCandidate, ...]
    exclusions: tuple[CandidateExclusion, ...]
    reasons: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class OracleTable:
    policy: OraclePolicy
    entries: tuple[OracleEntry, ...]

    def by_instance(self) -> dict[str, OracleEntry]:
        return {entry.instance: entry for entry in self.entries}


def _dotted_value(row: Mapping[str, Any], field: str) -> Any:
    current: Any = row
    for part in field.split("."):
        if not isinstance(current, Mapping) or part not in current:
            return None
        current = current[part]
    return current


def _exact_success(row: Mapping[str, Any]) -> bool:
    if row.get("state") != "solved" or row.get("validated") is not True:
        return False
    check = row.get("reference_check")
    if check in _REFERENCE_SUCCESS:
        return True
    if check == "not_checkable":
        evidence = row.get("optimality_evidence")
        return (
            row.get("public_status") == "valid_optimal"
            and isinstance(evidence, Mapping)
            and evidence.get("independently_verified_optimal") is True
        )
    return False


def _candidate(row: Mapping[str, Any], policy: OraclePolicy) -> tuple[OracleCandidate | None, CandidateExclusion | None]:
    run_id = str(row.get("run_id") or "")
    backend = str(row.get("backend") or "")
    if not run_id:
        raise ValueError("every oracle row must provide a non-empty run_id")
    if not backend:
        raise ValueError(f"run {run_id!r} has no backend")
    if backend == "@auto" and not policy.include_auto_backend:
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.AUTO_BACKEND, "automatic policy rows are excluded from the direct-run oracle")
    if row.get("state") != "solved":
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NOT_SOLVED, f"state={row.get('state')!r}")
    if row.get("validated") is not True:
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NOT_VALIDATED, "independent validation did not pass")
    if not _exact_success(row):
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NOT_EXACT_SUCCESS, f"reference_check={row.get('reference_check')!r}, public_status={row.get('public_status')!r}")
    raw_objective = row.get("objective")
    if raw_objective is None:
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.MISSING_OBJECTIVE, "eligible exact run has no objective")
    try:
        objective = float(raw_objective)
    except (TypeError, ValueError):
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NONFINITE_OBJECTIVE, "objective is not numeric")
    if not isfinite(objective):
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NONFINITE_OBJECTIVE, "objective is not finite")
    raw_runtime = _dotted_value(row, policy.runtime_field)
    if raw_runtime is None:
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.MISSING_RUNTIME, f"runtime field {policy.runtime_field!r} is missing")
    try:
        runtime = float(raw_runtime)
    except (TypeError, ValueError):
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NONFINITE_RUNTIME, "runtime is not numeric")
    if not isfinite(runtime):
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NONFINITE_RUNTIME, "runtime is not finite")
    if runtime < 0:
        return None, CandidateExclusion(run_id, backend, CandidateExclusionReason.NEGATIVE_RUNTIME, "runtime must be non-negative")
    return OracleCandidate(run_id, backend, objective, runtime), None


def _same_objective(a: float, b: float, policy: OraclePolicy) -> bool:
    return isclose(a, b, abs_tol=policy.objective_atol, rel_tol=policy.objective_rtol)


def _identity_value(row: Mapping[str, Any], field: str) -> str | None:
    raw = row.get(field)
    if raw is None:
        return None
    text = str(raw).strip()
    return text or None


def _valid_sha256(value: str | None) -> bool:
    return value is not None and len(value) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in value)


def build_oracle_table(rows: Iterable[Mapping[str, Any]], policy: OraclePolicy | None = None) -> OracleTable:
    policy = OraclePolicy() if policy is None else policy
    data = tuple(rows)
    if not data:
        raise ValueError("oracle construction requires at least one benchmark row")
    run_ids = [str(row.get("run_id") or "") for row in data]
    if any(not run_id for run_id in run_ids):
        raise ValueError("every oracle row must provide a non-empty run_id")
    if len(set(run_ids)) != len(run_ids):
        raise ValueError("run_id values must be unique")

    grouped: dict[str, list[Mapping[str, Any]]] = {}
    for row in data:
        instance = str(row.get("instance") or "")
        if not instance:
            raise ValueError(f"run {row.get('run_id')!r} has no instance")
        grouped.setdefault(instance, []).append(row)

    entries: list[OracleEntry] = []
    for instance in sorted(grouped):
        group = grouped[instance]
        protocols = {_identity_value(r, "protocol_id") for r in group}
        environments = {_identity_value(r, "environment_id") for r in group}
        instance_hashes = {_identity_value(r, "instance_sha256") for r in group}
        status: OracleStatus | None = None
        reasons: list[str] = []
        if policy.require_same_protocol and (len(protocols) != 1 or None in protocols):
            status = OracleStatus.PROTOCOL_MISMATCH
            reasons.append("candidate rows must provide one identical non-empty protocol_id")
        if status is None and policy.require_same_environment and (len(environments) != 1 or None in environments):
            status = OracleStatus.ENVIRONMENT_MISMATCH
            reasons.append("candidate rows must provide one identical non-empty environment_id")
        if status is None and policy.require_same_instance_sha256:
            if len(instance_hashes) != 1 or None in instance_hashes or not all(_valid_sha256(x) for x in instance_hashes):
                status = OracleStatus.INSTANCE_IDENTITY_MISMATCH
                reasons.append("candidate rows must provide one identical valid instance_sha256")

        candidates: list[OracleCandidate] = []
        exclusions: list[CandidateExclusion] = []
        for row in sorted(group, key=lambda r: str(r.get("run_id"))):
            candidate, exclusion = _candidate(row, policy)
            if candidate is not None:
                candidates.append(candidate)
            if exclusion is not None:
                exclusions.append(exclusion)

        oracle_value: float | None = None
        oracle_candidates: list[OracleCandidate] = []
        fastest_runtime: float | None = None
        if status is None:
            if not candidates:
                status = OracleStatus.NO_ELIGIBLE_CANDIDATE
                reasons.append("no validated exact-success row with finite objective and runtime")
            else:
                values = [c.objective for c in candidates]
                oracle_value = min(values) if policy.objective_sense is ObjectiveSense.MINIMIZE else max(values)
                oracle_candidates = [c for c in candidates if _same_objective(c.objective, oracle_value, policy)]
                fastest_runtime = min(c.runtime_s for c in oracle_candidates)
                if policy.runtime_tie_breaker:
                    oracle_candidates = [
                        c for c in oracle_candidates
                        if isclose(c.runtime_s, fastest_runtime, abs_tol=policy.runtime_tolerance_s, rel_tol=0.0)
                    ]
                status = OracleStatus.AVAILABLE

        entries.append(OracleEntry(
            instance=instance,
            instance_sha256=next(iter(instance_hashes)) if len(instance_hashes) == 1 else None,
            protocol_id=next(iter(protocols)) if len(protocols) == 1 else None,
            environment_id=next(iter(environments)) if len(environments) == 1 else None,
            status=status,
            objective_sense=policy.objective_sense,
            oracle_objective=oracle_value,
            oracle_run_ids=tuple(sorted(c.run_id for c in oracle_candidates)),
            eligible_run_ids=tuple(sorted(c.run_id for c in candidates)),
            fastest_oracle_runtime_s=fastest_runtime,
            candidates=tuple(candidates),
            exclusions=tuple(exclusions),
            reasons=tuple(reasons),
        ))
    return OracleTable(policy=policy, entries=tuple(entries))
