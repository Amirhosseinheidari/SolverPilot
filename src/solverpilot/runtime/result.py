from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

import numpy as np

from solverpilot._immutability import deep_freeze, readonly_array
from solverpilot.plan import SolvePlan
from solverpilot.trace import SolveTrace
from solverpilot.validate import PublicStatus, ValidationReport

if TYPE_CHECKING:
    from solverpilot.diagnose import InfeasibilityReport


@dataclass(frozen=True, slots=True)
class OptimalityEvidence:
    """Trust metadata for an apparent optimal termination.

    ``VALID_OPTIMAL`` remains a compatibility status meaning the backend reported
    optimality and the primal candidate passed canonical validation.  It does not
    imply that SolverPilot independently reconstructed a complete optimality proof.
    """

    backend_reported_optimal: bool = False
    primal_validated: bool = False
    dual_verified: bool = False
    gap_verified: bool = False
    certificate_verified: bool = False

    @property
    def independently_verified_optimal(self) -> bool:
        return bool(self.primal_validated and self.dual_verified and (self.gap_verified or self.certificate_verified))


@dataclass(frozen=True, slots=True)
class SolveResult:
    status: PublicStatus
    x: np.ndarray | None
    objective: float | None
    backend_status: str
    validation: ValidationReport | None
    trace: SolveTrace
    raw_statistics: dict[str, object] | None = None
    plan: SolvePlan | None = None
    session_revision: int | None = None
    diagnostics: "InfeasibilityReport | None" = None
    _optimality_evidence: OptimalityEvidence = field(init=False, repr=False, compare=False)

    def __post_init__(self) -> None:
        if self.x is not None:
            object.__setattr__(self, "x", readonly_array(self.x, dtype=np.float64))
        frozen_raw = None if self.raw_statistics is None else deep_freeze(self.raw_statistics)
        object.__setattr__(self, "raw_statistics", frozen_raw)

        raw = frozen_raw or {}
        trust = raw.get("solverpilot_trust", {})
        if not isinstance(trust, Mapping):
            trust = {}
        object.__setattr__(self, "_optimality_evidence", OptimalityEvidence(
            backend_reported_optimal=bool(trust.get("backend_reported_optimal", self.backend_status == "optimal")),
            primal_validated=bool(trust.get("primal_validated", self.validation is not None and self.validation.valid)),
            dual_verified=bool(trust.get("dual_verified", False)),
            gap_verified=bool(trust.get("gap_verified", False)),
            certificate_verified=bool(trust.get("certificate_verified", False)),
        ))

    @property
    def optimality_evidence(self) -> OptimalityEvidence:
        return self._optimality_evidence
