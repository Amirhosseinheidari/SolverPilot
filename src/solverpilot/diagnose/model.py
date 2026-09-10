from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np


class EvidenceKind(str, Enum):
    STATIC_CONTRADICTION = "static_contradiction"
    NATIVE_IIS = "native_iis"
    ELASTIC_RELAXATION = "elastic_relaxation"
    DELETION_FILTER = "deletion_filter"


class ViolationKind(str, Enum):
    ROW_LOWER = "row_lower"
    ROW_UPPER = "row_upper"
    VARIABLE_LOWER = "variable_lower"
    VARIABLE_UPPER = "variable_upper"
    INTEGER_DOMAIN = "integer_domain"
    EMPTY_ROW = "empty_row"


@dataclass(frozen=True, slots=True)
class DiagnosticIssue:
    kind: ViolationKind
    index: int
    message: str
    amount: float | None = None


@dataclass(frozen=True, slots=True)
class RelaxationViolation:
    kind: ViolationKind
    index: int
    amount: float
    normalized_amount: float


@dataclass(frozen=True, slots=True)
class ElasticRelaxationResult:
    feasible: bool
    objective: float | None
    x: np.ndarray | None
    violations: tuple[RelaxationViolation, ...]
    solver_status: str
    message: str


@dataclass(frozen=True, slots=True)
class NormalizedIISEvidence:
    backend: str
    valid: bool
    row_indices: tuple[int, ...] = ()
    variable_indices: tuple[int, ...] = ()
    constraint_names: tuple[str, ...] = ()
    variable_names: tuple[str, ...] = ()
    irreducible: bool | None = None
    raw_type: str | None = None


@dataclass(frozen=True, slots=True)
class ConflictAtom:
    kind: ViolationKind
    index: int


@dataclass(frozen=True, slots=True)
class ConflictSetResult:
    atoms: tuple[ConflictAtom, ...]
    irreducible: bool
    complete: bool
    checks: int
    solver_status: str
    message: str


@dataclass(frozen=True, slots=True)
class InfeasibilityReport:
    confirmed_infeasible: bool
    static_issues: tuple[DiagnosticIssue, ...]
    iis: NormalizedIISEvidence | None
    elastic: ElasticRelaxationResult | None
    conflict: ConflictSetResult | None
    evidence_order: tuple[EvidenceKind, ...]
    warnings: tuple[str, ...] = ()
