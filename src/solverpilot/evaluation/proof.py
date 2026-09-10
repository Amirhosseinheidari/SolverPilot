from __future__ import annotations

from dataclasses import asdict, dataclass
from enum import IntEnum
from typing import Any, Mapping


class ProofLevel(IntEnum):
    NONE = 0
    SOLVER_ASSERTED = 1
    PRIMAL_VALIDATED = 2
    BOUND_VERIFIED = 3
    INDEPENDENT_CERTIFICATE = 4


@dataclass(frozen=True, slots=True)
class ProofAssessment:
    level: ProofLevel
    backend_reported_optimal: bool
    primal_validated: bool
    dual_verified: bool
    gap_verified: bool
    certificate_verified: bool
    independently_verified_optimal: bool
    reasons: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        payload=asdict(self); payload["level"]=int(self.level); payload["level_name"]=self.level.name.lower(); return payload


def _from_flags(*, backend_reported_optimal: bool, primal_validated: bool, dual_verified: bool, gap_verified: bool, certificate_verified: bool) -> ProofAssessment:
    independently=bool(primal_validated and dual_verified and (gap_verified or certificate_verified)); reasons=[]
    if certificate_verified and independently: level=ProofLevel.INDEPENDENT_CERTIFICATE
    elif primal_validated and dual_verified and gap_verified: level=ProofLevel.BOUND_VERIFIED
    elif primal_validated: level=ProofLevel.PRIMAL_VALIDATED
    elif backend_reported_optimal: level=ProofLevel.SOLVER_ASSERTED
    else: level=ProofLevel.NONE
    if backend_reported_optimal and not primal_validated: reasons.append("backend optimality claim lacks an independently validated primal candidate")
    if primal_validated and not independently: reasons.append("primal feasibility is validated, but complete optimality proof is not independently reconstructed")
    return ProofAssessment(level,backend_reported_optimal,primal_validated,dual_verified,gap_verified,certificate_verified,independently,tuple(reasons))


def assess_solve_result(result: object) -> ProofAssessment:
    evidence=getattr(result,"optimality_evidence",None)
    if evidence is None: return _from_flags(backend_reported_optimal=False,primal_validated=False,dual_verified=False,gap_verified=False,certificate_verified=False)
    return _from_flags(backend_reported_optimal=bool(getattr(evidence,"backend_reported_optimal",False)),primal_validated=bool(getattr(evidence,"primal_validated",False)),dual_verified=bool(getattr(evidence,"dual_verified",False)),gap_verified=bool(getattr(evidence,"gap_verified",False)),certificate_verified=bool(getattr(evidence,"certificate_verified",False)))


def assess_benchmark_row(row: Mapping[str, Any]) -> ProofAssessment:
    evidence=row.get("optimality_evidence")
    if isinstance(evidence,Mapping):
        return _from_flags(backend_reported_optimal=bool(evidence.get("backend_reported_optimal",row.get("public_status")=="valid_optimal")),primal_validated=bool(evidence.get("primal_validated",row.get("validated") is True)),dual_verified=bool(evidence.get("dual_verified",False)),gap_verified=bool(evidence.get("gap_verified",False)),certificate_verified=bool(evidence.get("certificate_verified",False)))
    return _from_flags(backend_reported_optimal=row.get("public_status")=="valid_optimal",primal_validated=row.get("validated") is True,dual_verified=False,gap_verified=False,certificate_verified=False)
