from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class PublicStatus(str, Enum):
    VALID_OPTIMAL = "valid_optimal"
    VALID_FEASIBLE = "valid_feasible"
    FEASIBLE_LIMIT = "feasible_limit"
    INFEASIBLE = "infeasible"
    UNBOUNDED = "unbounded"
    INFEASIBLE_OR_UNBOUNDED = "infeasible_or_unbounded"
    INVALID_SOLUTION = "invalid_solution"
    ERROR = "error"
    UNKNOWN = "unknown"


@dataclass(frozen=True, slots=True)
class ValidationTolerances:
    feasibility: float = 1e-7
    integrality: float = 1e-6
    objective_abs: float = 1e-7
    objective_rel: float = 1e-7

    def __post_init__(self) -> None:
        for name in ("feasibility", "integrality", "objective_abs", "objective_rel"):
            raw = getattr(self, name)
            if isinstance(raw, bool) or type(raw) not in (int, float):
                raise ValueError(f"{name} tolerance must be a finite number")
            value = float(raw)
            if not np.isfinite(value) or value < 0:
                raise ValueError(f"{name} tolerance must be finite and non-negative")
            object.__setattr__(self, name, value)


@dataclass(frozen=True, slots=True)
class ValidationReport:
    valid: bool
    max_bound_violation: float
    max_constraint_violation: float
    max_integrality_violation: float | None
    objective_recomputed: float | None
    objective_reported: float | None
    objective_difference: float | None
    objective_consistent: bool | None
    warnings: tuple[str, ...] = ()


@dataclass(frozen=True, slots=True)
class CandidateSolution:
    x: np.ndarray
    objective_reported: float | None = None
