from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import math

from solverpilot.inspect import ProblemFingerprint


class SolveIntent(str, Enum):
    PROVE_OPTIMAL = "prove_optimal"
    BEST_FEASIBLE_UNDER_BUDGET = "best_feasible_under_budget"
    FAST_FEASIBLE = "fast_feasible"
    BALANCED = "balanced"
    HIGH_ACCURACY = "high_accuracy"
    LOW_MEMORY = "low_memory"


class HealthPolicy(str, Enum):
    """How planner evidence from explicit backend probes is used.

    ``IGNORE`` preserves the capability-only planner. ``PREFER_HEALTHY`` excludes
    backends that were explicitly observed unhealthy and gives a small tie-break to
    healthy backends while leaving unprobed backends eligible. ``REQUIRE_HEALTHY``
    admits only backends with a successful active smoke probe.
    """

    IGNORE = "ignore"
    PREFER_HEALTHY = "prefer_healthy"
    REQUIRE_HEALTHY = "require_healthy"


@dataclass(frozen=True, slots=True)
class SolveBudget:
    wall_time_s: float | None = None
    memory_mb: int | None = None
    threads: int | None = None

    def __post_init__(self) -> None:
        if self.wall_time_s is not None:
            if isinstance(self.wall_time_s, bool) or type(self.wall_time_s) not in (int, float):
                raise ValueError("wall_time_s must be a finite positive number")
            value = float(self.wall_time_s)
            if not math.isfinite(value) or value <= 0:
                raise ValueError("wall_time_s must be a finite positive number")
            object.__setattr__(self, "wall_time_s", value)
        for name in ("memory_mb", "threads"):
            raw = getattr(self, name)
            if raw is None:
                continue
            if isinstance(raw, bool) or type(raw) is not int or raw <= 0:
                raise ValueError(f"{name} must be a positive integer")


@dataclass(frozen=True, slots=True)
class PlannerContext:
    previous_backend: str | None = None
    session_revision: int | None = None


@dataclass(frozen=True, slots=True)
class CandidatePlan:
    backend: str
    score: float
    rationale: tuple[str, ...]
    health_status: str | None = None


@dataclass(frozen=True, slots=True)
class SolvePlan:
    selected_backend: str
    candidates: tuple[CandidatePlan, ...]
    intent: SolveIntent
    budget: SolveBudget
    fingerprint: ProblemFingerprint
    strategy: str
    rationale: tuple[str, ...]
    evidence_level: str = "capability_only"
    health_policy: HealthPolicy = HealthPolicy.IGNORE
