from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any


TRACE_SCHEMA_VERSION = "0.3"


@dataclass(frozen=True, slots=True)
class PhaseTimings:
    inspect_s: float = 0.0
    plan_s: float = 0.0
    backend_build_s: float = 0.0
    solve_s: float = 0.0
    validate_s: float = 0.0
    diagnose_s: float = 0.0
    total_s: float = 0.0


@dataclass(frozen=True, slots=True)
class SolveTrace:
    problem_structural_hash: str
    problem_data_hash: str
    problem_class: str
    backend: str | None = None
    backend_version: str | None = None
    parameters: dict[str, Any] = field(default_factory=dict)
    fingerprint: dict[str, Any] | None = None
    planner_selected_backend: str | None = None
    planner_evidence_level: str | None = None
    planner_health_policy: str | None = None
    transformations: tuple[str, ...] = ()
    seed: int | None = None
    threads: int | None = None
    timings: PhaseTimings = field(default_factory=PhaseTimings)
    termination: str | None = None
    validation_valid: bool | None = None
    reuse_applied: bool | None = None
    reuse_mode: str | None = None
    warnings: tuple[str, ...] = ()
    schema_version: str = TRACE_SCHEMA_VERSION
    created_at_utc: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
