from __future__ import annotations

from dataclasses import asdict
from typing import Any

from solverpilot.runtime import SolveResult
from solverpilot.validate import PublicStatus

from .model import ClaimDisposition, ClaimKind, ExplanationClaim, ExplanationReport


_STATUS_EXPLANATIONS: dict[PublicStatus, str] = {
    PublicStatus.VALID_OPTIMAL: (
        "The backend reported optimality and the returned primal candidate passed SolverPilot's canonical validation. "
        "This status alone is not an independent proof of optimality; consult the optimality evidence in this report."
    ),
    PublicStatus.VALID_FEASIBLE: (
        "The returned candidate passed SolverPilot's canonical feasibility validation. "
        "Feasibility does not establish optimality."
    ),
    PublicStatus.FEASIBLE_LIMIT: (
        "A candidate passed canonical validation, but the backend terminated at a configured or solver limit. "
        "The result must not be presented as proven optimal."
    ),
    PublicStatus.INFEASIBLE: (
        "The backend reported the model infeasible. Additional SolverPilot diagnostics, when present, are reported separately; "
        "the public status alone is not an independently reconstructed universal infeasibility proof."
    ),
    PublicStatus.UNBOUNDED: (
        "The backend reported the model unbounded. SolverPilot does not infer an independent unboundedness proof from this status alone."
    ),
    PublicStatus.INFEASIBLE_OR_UNBOUNDED: (
        "The backend could not distinguish infeasibility from unboundedness. Neither conclusion should be stated separately without more evidence."
    ),
    PublicStatus.INVALID_SOLUTION: (
        "The backend returned a candidate that failed SolverPilot's independent canonical validation. "
        "The candidate must not be presented as a valid solution."
    ),
    PublicStatus.ERROR: (
        "The solve did not produce a validated normal result. No feasibility, optimality, infeasibility, or superiority claim follows from this status."
    ),
    PublicStatus.UNKNOWN: (
        "The outcome could not be mapped to a stronger public status. Additional evidence is required before making a solve-quality claim."
    ),
}


def explain_status(status: PublicStatus | str) -> str:
    if isinstance(status, str):
        status = PublicStatus(status)
    return _STATUS_EXPLANATIONS[status]


def _validation_payload(result: SolveResult) -> dict[str, Any] | None:
    report = result.validation
    if report is None:
        return None
    return {
        "valid": report.valid,
        "max_bound_violation": report.max_bound_violation,
        "max_constraint_violation": report.max_constraint_violation,
        "max_integrality_violation": report.max_integrality_violation,
        "objective_recomputed": report.objective_recomputed,
        "objective_reported": report.objective_reported,
        "objective_difference": report.objective_difference,
        "objective_consistent": report.objective_consistent,
        "warnings": tuple(report.warnings),
    }


def _optimality_payload(result: SolveResult) -> dict[str, bool]:
    evidence = result.optimality_evidence
    return {
        "backend_reported_optimal": evidence.backend_reported_optimal,
        "primal_validated": evidence.primal_validated,
        "dual_verified": evidence.dual_verified,
        "gap_verified": evidence.gap_verified,
        "certificate_verified": evidence.certificate_verified,
        "independently_verified_optimal": evidence.independently_verified_optimal,
    }


def _planner_payload(result: SolveResult) -> dict[str, Any] | None:
    plan = result.plan
    if plan is None:
        return None
    return {
        "selected_backend": plan.selected_backend,
        "intent": plan.intent.value,
        "strategy": plan.strategy,
        "evidence_level": plan.evidence_level,
        "health_policy": plan.health_policy.value,
        "rationale": tuple(plan.rationale),
        "candidate_count": len(plan.candidates),
    }


def _runtime_payload(result: SolveResult) -> dict[str, Any]:
    timings = result.trace.timings
    return {
        "reuse_applied": result.trace.reuse_applied,
        "reuse_mode": result.trace.reuse_mode,
        "total_s": timings.total_s,
        "solve_s": timings.solve_s,
        "validate_s": timings.validate_s,
        "diagnose_s": timings.diagnose_s,
    }


def _diagnostics_payload(result: SolveResult) -> dict[str, Any] | None:
    diagnostics = result.diagnostics
    if diagnostics is None:
        return None
    return {
        "confirmed_infeasible": diagnostics.confirmed_infeasible,
        "static_issue_count": len(diagnostics.static_issues),
        "iis_available": diagnostics.iis is not None,
        "iis_valid": bool(diagnostics.iis is not None and diagnostics.iis.valid),
        "elastic_available": diagnostics.elastic is not None,
        "conflict_available": diagnostics.conflict is not None,
        "conflict_irreducible": None if diagnostics.conflict is None else diagnostics.conflict.irreducible,
        "warnings": tuple(diagnostics.warnings),
    }


def _claims(result: SolveResult) -> tuple[ExplanationClaim, ...]:
    claims: list[ExplanationClaim] = []
    validation_valid = bool(result.validation is not None and result.validation.valid and result.x is not None)
    if validation_valid:
        claims.append(
            ExplanationClaim(
                claim_id="solution.feasibility",
                kind=ClaimKind.FEASIBILITY,
                disposition=ClaimDisposition.SUPPORTED,
                statement="The returned candidate passed SolverPilot's canonical validation.",
                rationale="A primal candidate exists and ValidationReport.valid is true.",
                evidence_refs=("result.validation",),
            )
        )
    else:
        claims.append(
            ExplanationClaim(
                claim_id="solution.feasibility",
                kind=ClaimKind.FEASIBILITY,
                disposition=ClaimDisposition.BLOCKED,
                statement="A validated feasible candidate cannot be claimed from this result.",
                rationale="No primal candidate passed SolverPilot's canonical validation.",
                evidence_refs=("result.validation",),
            )
        )

    evidence = result.optimality_evidence
    if evidence.independently_verified_optimal:
        claims.append(
            ExplanationClaim(
                claim_id="solution.optimality",
                kind=ClaimKind.OPTIMALITY,
                disposition=ClaimDisposition.SUPPORTED,
                statement="Optimality satisfied SolverPilot's available independent optimality checks.",
                rationale="Primal validation, dual verification, and a verified gap or certificate are all present.",
                evidence_refs=("result.optimality_evidence",),
                qualifiers=("The scope of verification is limited to the checks represented by OptimalityEvidence.",),
            )
        )
    elif result.status is PublicStatus.VALID_OPTIMAL and evidence.backend_reported_optimal and evidence.primal_validated:
        claims.append(
            ExplanationClaim(
                claim_id="solution.optimality",
                kind=ClaimKind.OPTIMALITY,
                disposition=ClaimDisposition.QUALIFIED,
                statement="The backend reported optimality and the primal candidate passed canonical validation.",
                rationale="SolverPilot did not independently reconstruct all optimality conditions required by OptimalityEvidence.",
                evidence_refs=("result.optimality_evidence", "result.validation"),
                qualifiers=("Do not describe this as independently proven optimal.",),
            )
        )
    else:
        claims.append(
            ExplanationClaim(
                claim_id="solution.optimality",
                kind=ClaimKind.OPTIMALITY,
                disposition=ClaimDisposition.BLOCKED,
                statement="Optimality is not established by this result.",
                rationale="The result does not carry sufficient optimality evidence.",
                evidence_refs=("result.optimality_evidence",),
            )
        )

    if result.status is PublicStatus.INFEASIBLE:
        confirmed = bool(result.diagnostics is not None and result.diagnostics.confirmed_infeasible)
        claims.append(
            ExplanationClaim(
                claim_id="solution.infeasibility",
                kind=ClaimKind.INFEASIBILITY,
                disposition=ClaimDisposition.SUPPORTED if confirmed else ClaimDisposition.QUALIFIED,
                statement=(
                    "SolverPilot diagnostics additionally confirmed infeasibility for the supplied model."
                    if confirmed
                    else "The backend reported the supplied model infeasible."
                ),
                rationale=(
                    "InfeasibilityReport.confirmed_infeasible is true."
                    if confirmed
                    else "No independent SolverPilot diagnostic confirmation is attached to this SolveResult."
                ),
                evidence_refs=("result.diagnostics", "result.backend_status") if confirmed else ("result.backend_status",),
                qualifiers=(
                    "The claim is limited to the supplied mathematical model and data; it is not a statement about an external real-world process.",
                ),
            )
        )
    else:
        claims.append(
            ExplanationClaim(
                claim_id="solution.infeasibility",
                kind=ClaimKind.INFEASIBILITY,
                disposition=ClaimDisposition.BLOCKED,
                statement="Infeasibility is not established by this result.",
                rationale=f"Public status is {result.status.value!r}, not 'infeasible'.",
                evidence_refs=("result.status",),
            )
        )

    if result.status is PublicStatus.UNBOUNDED:
        claims.append(
            ExplanationClaim(
                claim_id="solution.unboundedness",
                kind=ClaimKind.UNBOUNDEDNESS,
                disposition=ClaimDisposition.QUALIFIED,
                statement="The backend reported the supplied model unbounded.",
                rationale="SolverPilot currently exposes no general independent unboundedness-proof field on SolveResult.",
                evidence_refs=("result.backend_status",),
                qualifiers=("Do not describe this as independently proven unbounded from SolveResult alone.",),
            )
        )
    else:
        claims.append(
            ExplanationClaim(
                claim_id="solution.unboundedness",
                kind=ClaimKind.UNBOUNDEDNESS,
                disposition=ClaimDisposition.BLOCKED,
                statement="Unboundedness is not established by this result.",
                rationale=f"Public status is {result.status.value!r}, not 'unbounded'.",
                evidence_refs=("result.status",),
            )
        )

    if result.plan is not None:
        claims.append(
            ExplanationClaim(
                claim_id="routing.selection",
                kind=ClaimKind.PLANNER_SELECTION,
                disposition=ClaimDisposition.SUPPORTED,
                statement=f"SolverPilot's planner selected backend {result.plan.selected_backend!r}.",
                rationale="The selection is recorded in SolvePlan.",
                evidence_refs=("result.plan",),
                qualifiers=("Planner selection is not, by itself, evidence that this backend is globally fastest or superior.",),
            )
        )

    if result.trace.reuse_applied is True:
        claims.append(
            ExplanationClaim(
                claim_id="runtime.reuse",
                kind=ClaimKind.REOPTIMIZATION,
                disposition=ClaimDisposition.SUPPORTED,
                statement="The recorded solve reused prior solver state.",
                rationale="SolveTrace.reuse_applied is true.",
                evidence_refs=("result.trace",),
                qualifiers=("State reuse does not by itself establish a performance improvement.",),
            )
        )

    return tuple(claims)


def _warnings(result: SolveResult) -> tuple[str, ...]:
    values: list[str] = []
    if result.validation is not None:
        values.extend(str(x) for x in result.validation.warnings)
    values.extend(str(x) for x in result.trace.warnings)
    if result.diagnostics is not None:
        values.extend(str(x) for x in result.diagnostics.warnings)
    # Preserve order while removing duplicates.
    return tuple(dict.fromkeys(values))


def explain_result(result: SolveResult) -> ExplanationReport:
    if not isinstance(result, SolveResult):
        raise TypeError("explain_result expects a SolveResult")
    optimality = _optimality_payload(result)
    independently_verified = optimality["independently_verified_optimal"]
    if result.status is PublicStatus.VALID_OPTIMAL:
        summary = (
            "A validated candidate was returned and optimality passed SolverPilot's available independent checks."
            if independently_verified
            else "A validated candidate was returned; the backend reported optimality, but complete independent optimality verification is not available."
        )
    elif result.status in {PublicStatus.VALID_FEASIBLE, PublicStatus.FEASIBLE_LIMIT}:
        summary = "A validated feasible candidate was returned without an independent optimality claim."
    elif result.status is PublicStatus.INFEASIBLE:
        summary = (
            "The model was reported infeasible and SolverPilot diagnostics additionally confirmed infeasibility."
            if result.diagnostics is not None and result.diagnostics.confirmed_infeasible
            else "The backend reported the supplied model infeasible; no additional SolverPilot confirmation is attached."
        )
    elif result.status is PublicStatus.INVALID_SOLUTION:
        summary = "The backend returned a candidate that failed SolverPilot's canonical validation."
    else:
        summary = f"The solve terminated with public status {result.status.value!r}."

    return ExplanationReport(
        status=result.status.value,
        summary=summary,
        status_explanation=explain_status(result.status),
        problem_class=result.trace.problem_class,
        backend=result.trace.backend,
        backend_version=result.trace.backend_version,
        objective=result.objective,
        validation=_validation_payload(result),
        optimality=optimality,
        planner=_planner_payload(result),
        runtime=_runtime_payload(result),
        diagnostics=_diagnostics_payload(result),
        claims=_claims(result),
        warnings=_warnings(result),
    )
