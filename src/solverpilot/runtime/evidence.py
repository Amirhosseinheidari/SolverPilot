"""Portable core LP/QP evidence envelopes; recorded claims are not re-certified on load."""

import json
from math import isfinite
from pathlib import Path

from .manifest import (
    _replay_payload, _seal, _summary_payload, _verify_integrity, json_value, run_manifest,
)
from .result import OptimalityEvidence, SolveResult


def evidence_bundle(problem, result, *, include_model=False, include_raw_statistics=False):
    """Link one core execution's summary, explanation and replay manifest.

    Default exports exclude model arrays, parameters, candidate vectors, raw
    backend statistics and free-text warnings. Objective/timing/identity and
    structured evidence remain visible. Full model export is explicit.
    """
    from solverpilot.reporting import explain_result

    if not isinstance(result, SolveResult):
        raise TypeError("evidence bundles currently require a core SolveResult")
    manifest = run_manifest(problem, result, include_model=include_model,
                            include_raw_statistics=include_raw_statistics)
    try:
        report = explain_result(result).to_dict()
    except ValueError:
        # Invalid solver candidates can carry nonfinite residuals. Preserve a
        # failed explanation explicitly instead of coercing it into a valid
        # claim or losing the entire execution export. No exception text leaks.
        summary = _summary_payload(result)
        report = {
            "schema_version": "unavailable", "status": summary["status"],
            "summary": "Structured explanation unavailable; consult the recorded execution status.",
            "status_explanation": "No additional report claim is established.",
            "problem_class": result.trace.problem_class, "backend": summary["backend"],
            "backend_version": result.trace.backend_version, "objective": summary["objective"],
            "optimality": {}, "claims": [], "validation": None, "planner": None,
            "diagnostics": None, "runtime": {
                "execution_id": result.execution_id, "source_sha256": result.trace.source_sha256,
                "reuse_applied": None, "total_s": None, "solve_s": None,
                "validate_s": None, "diagnose_s": None,
            },
        }
    explanation = {key: report[key] for key in (
        "schema_version", "status", "summary", "status_explanation", "problem_class",
        "backend", "backend_version", "objective", "optimality", "claims",
    )}
    allowed = {
        "validation": ("valid", "max_bound_violation", "max_constraint_violation",
                       "max_integrality_violation", "objective_recomputed", "objective_reported",
                       "objective_difference", "objective_consistent"),
        "planner": ("selected_backend", "intent", "health_policy", "candidate_count"),
        "runtime": ("execution_id", "source_sha256", "reuse_applied", "total_s", "solve_s",
                    "validate_s", "diagnose_s"),
        "diagnostics": ("confirmed_infeasible", "static_issue_count", "iis_available", "iis_valid",
                        "elastic_available", "conflict_available", "conflict_irreducible"),
    }
    for section, keys in allowed.items():
        explanation[section] = None if report[section] is None else {k: report[section][k] for k in keys}
    return _seal({
        "schema": "solverpilot.evidence.v1",
        "execution_id": result.execution_id,
        "problem_data_hash": problem.data_hash,
        "summary": _summary_payload(result),
        "explanation": explanation,
        "run": manifest,
        "omissions": ["candidate_vectors", "free_text_warnings"],
        "verification_scope": "recorded execution evidence; load checks integrity, not mathematical validity",
    })


def _boolean(value, name, *, nullable=False):
    if type(value) is not bool and not (nullable and value is None):
        raise ValueError(f"evidence {name} must be a boolean" + (" or null" if nullable else ""))
    return value


def _number(value, name, *, nullable=False, nonnegative=False):
    if nullable and value is None:
        return
    if (type(value) not in (int, float) or not isfinite(value)
            or (nonnegative and value < 0)):
        raise ValueError(f"evidence {name} must be a finite number")


def _verify_summary_types(summary):
    # Python considers True == 1, including inside dict equality. Check both
    # copies before linking them so a resealed nested summary cannot exploit it.
    _boolean(summary["feasible"], "summary.feasible")
    _boolean(summary["within_budget"], "summary.within_budget", nullable=True)
    _number(summary["objective"], "summary.objective", nullable=True)
    for key in ("requested_time_s", "elapsed_s"):
        _number(summary[key], f"summary.{key}", nullable=True, nonnegative=True)
    requested, elapsed, within = (summary[key] for key in ("requested_time_s", "elapsed_s", "within_budget"))
    # Core solve records use <=, including an elapsed time exactly at the budget.
    # Null values remain unavailable: loading cannot invent missing observations.
    if requested is not None and elapsed is not None and within is not None and within != (elapsed <= requested):
        raise ValueError("evidence summary.within_budget disagrees with recorded times")


def _verify_recorded_explanation(summary, explanation):
    """Check agreement between recorded fields, not whether a solver proof is true."""
    from solverpilot.reporting.model import ExplanationClaim, ExplanationReport, REPORT_SCHEMA_VERSION
    from solverpilot.validate import PublicStatus

    status = PublicStatus(summary["status"])
    feasible = _boolean(summary["feasible"], "summary.feasible")
    if summary["optimality"] not in {"not_established", "solver_reported", "independent_numerical_bound"}:
        raise ValueError("unsupported core evidence summary optimality")
    if not feasible and summary["optimality"] != "not_established":
        raise ValueError("evidence summary claims optimality without feasibility")
    if status in {PublicStatus.VALID_OPTIMAL, PublicStatus.VALID_FEASIBLE, PublicStatus.FEASIBLE_LIMIT} and not feasible:
        raise ValueError("evidence validated status disagrees with feasibility")
    if status is PublicStatus.INVALID_SOLUTION and feasible:
        raise ValueError("evidence invalid status disagrees with feasibility")

    schema = explanation["schema_version"]
    if schema not in {REPORT_SCHEMA_VERSION, "unavailable"}:
        raise ValueError("unsupported evidence explanation schema")
    if not isinstance(explanation["claims"], list):
        raise ValueError("evidence explanation claims must be a list")
    claims = tuple(ExplanationClaim(**claim) for claim in explanation["claims"])
    # The normal report model also validates field types, claim IDs, evidence
    # references and supported-claim prerequisites. No serialized prose is
    # regenerated or matched, preserving existing v1 bundles and their wording.
    ExplanationReport(**{**explanation, "claims": claims, "schema_version": REPORT_SCHEMA_VERSION})
    runtime = explanation["runtime"]
    if schema == "unavailable":
        if (explanation["optimality"] != {} or claims
                or any(explanation[key] is not None for key in ("validation", "planner", "diagnostics"))
                or any(runtime[key] is not None for key in (
                    "reuse_applied", "total_s", "solve_s", "validate_s", "diagnose_s"))):
            raise ValueError("unavailable evidence explanation must contain no explanation claims or evidence")
        return

    validation = explanation["validation"]
    validated = False if validation is None else _boolean(validation["valid"], "validation.valid")
    if validated != feasible:
        raise ValueError("evidence validation disagrees with summary feasibility")
    if validation is not None:
        _boolean(validation["objective_consistent"], "validation.objective_consistent", nullable=True)
        for key in ("max_bound_violation", "max_constraint_violation"):
            _number(validation[key], f"validation.{key}", nonnegative=True)
        _number(validation["max_integrality_violation"], "validation.max_integrality_violation",
                nullable=True, nonnegative=True)
        for key in ("objective_recomputed", "objective_reported", "objective_difference"):
            _number(validation[key], f"validation.{key}", nullable=True)
        if validated and (validation["objective_recomputed"] is None or validation["objective_consistent"] is False):
            raise ValueError("evidence valid candidate contradicts its objective validation")
        if validation["objective_recomputed"] is not None and validation["objective_recomputed"] != summary["objective"]:
            raise ValueError("evidence validation objective disagrees with summary")
        if validation["objective_reported"] is None:
            if validation["objective_difference"] is not None or validation["objective_consistent"] is not None:
                raise ValueError("evidence objective comparison has no reported objective")
        elif validation["objective_recomputed"] is not None:
            difference = abs(validation["objective_recomputed"] - validation["objective_reported"])
            if (validation["objective_difference"] != difference or validation["objective_consistent"] is None
                    or (difference == 0 and validation["objective_consistent"] is not True)):
                raise ValueError("evidence objective comparison disagrees with recorded objective values")

    optimality = explanation["optimality"]
    if set(optimality) != {"backend_reported_optimal", "primal_validated", "dual_verified", "gap_verified",
                          "certificate_verified", "independently_verified_optimal"}:
        raise ValueError("unsupported evidence optimality flags")
    evidence = OptimalityEvidence(**{
        key: _boolean(optimality[key], f"optimality.{key}") for key in (
            "backend_reported_optimal", "primal_validated", "dual_verified", "gap_verified", "certificate_verified")
    })
    independent = _boolean(optimality["independently_verified_optimal"], "optimality.independently_verified_optimal")
    if evidence.primal_validated != feasible or independent != evidence.independently_verified_optimal:
        raise ValueError("evidence optimality flags disagree with their prerequisites")
    expected_optimality = ("independent_numerical_bound" if independent else
                           "solver_reported" if feasible and evidence.backend_reported_optimal else "not_established")
    if summary["optimality"] != expected_optimality:
        raise ValueError("evidence explanation optimality disagrees with summary")
    if status is PublicStatus.VALID_OPTIMAL and not evidence.backend_reported_optimal:
        raise ValueError("evidence optimal status lacks backend-reported optimality")

    diagnostics = explanation["diagnostics"]
    confirmed = False
    if diagnostics is not None:
        for key in ("confirmed_infeasible", "iis_available", "iis_valid", "elastic_available", "conflict_available"):
            _boolean(diagnostics[key], f"diagnostics.{key}")
        _boolean(diagnostics["conflict_irreducible"], "diagnostics.conflict_irreducible", nullable=True)
        if (diagnostics["iis_valid"] and not diagnostics["iis_available"]
                or diagnostics["conflict_irreducible"] is not None and not diagnostics["conflict_available"]):
            raise ValueError("evidence diagnostic flags disagree with available evidence")
        confirmed = diagnostics["confirmed_infeasible"]
        if confirmed and feasible:
            raise ValueError("evidence infeasibility diagnosis contradicts a validated candidate")
    _boolean(runtime["reuse_applied"], "runtime.reuse_applied", nullable=True)
    for key in ("total_s", "solve_s", "validate_s", "diagnose_s"):
        _number(runtime[key], f"runtime.{key}", nonnegative=True)

    expected_claims = {
        "solution.feasibility": ("feasibility", "supported" if feasible else "blocked"),
        "solution.optimality": ("optimality", "supported" if independent else
            "qualified" if status is PublicStatus.VALID_OPTIMAL and evidence.backend_reported_optimal and feasible else "blocked"),
        "solution.infeasibility": ("infeasibility", ("supported" if confirmed else "qualified")
            if status is PublicStatus.INFEASIBLE else "blocked"),
        "solution.unboundedness": ("unboundedness", "qualified" if status is PublicStatus.UNBOUNDED else "blocked"),
    }
    if explanation["planner"] is not None:
        if explanation["planner"]["selected_backend"] != summary["backend"]:
            raise ValueError("evidence planner selection disagrees with backend")
        expected_claims["routing.selection"] = ("planner_selection", "supported")
    if runtime["reuse_applied"] is True:
        expected_claims["runtime.reuse"] = ("reoptimization", "supported")
    actual_claims = {claim.claim_id: (claim.kind.value, claim.disposition.value) for claim in claims}
    if actual_claims != expected_claims:
        raise ValueError("evidence explanation claims disagree with recorded evidence")


def verify_evidence_bundle(payload):
    """Check recorded integrity, linkage and consistency, never authenticate or re-certify claims."""
    try:
        return _verify_evidence_bundle(payload)
    except (KeyError, TypeError, AttributeError, OverflowError) as exc:
        raise ValueError("malformed evidence bundle") from exc


def _verify_evidence_bundle(payload):
    if not isinstance(payload, dict) or payload.get("schema") != "solverpilot.evidence.v1":
        raise ValueError("unsupported evidence bundle schema")
    _verify_integrity(payload)
    run = payload["run"]
    _verify_integrity(run)
    if run.get("schema") != "solverpilot.run.v2":
        raise ValueError("unsupported nested run schema")
    identifier = payload["execution_id"]
    if not isinstance(identifier, str) or not identifier.strip() or any(
        value != identifier for value in (
            run["execution_id"], payload["summary"]["run_id"], run["summary"]["run_id"],
            payload["explanation"]["runtime"]["execution_id"],
        )
    ):
        raise ValueError("evidence execution identity mismatch")
    if any(value != payload["problem_data_hash"] for value in (
        run["problem_data_hash"], payload["summary"]["problem_data_hash"],
        run["summary"]["problem_data_hash"],
    )):
        raise ValueError("evidence problem identity mismatch")
    _verify_summary_types(payload["summary"])
    _verify_summary_types(run["summary"])
    if payload["summary"] != run["summary"]:
        raise ValueError("evidence summaries disagree")
    if any(run[key] != payload["summary"][key] for key in ("status", "backend")):
        raise ValueError("evidence run disagrees with summary")
    if payload["explanation"]["runtime"]["source_sha256"] != run["code_identity"]["source_sha256"]:
        raise ValueError("evidence source identity mismatch")
    if any(payload["explanation"][key] != payload["summary"][key]
            for key in ("status", "objective", "backend")):
        raise ValueError("evidence explanation disagrees with summary")
    _verify_recorded_explanation(payload["summary"], payload["explanation"])
    return payload


def save_evidence_bundle(path, problem, result, **options):
    payload = evidence_bundle(problem, result, **options)
    Path(path).write_text(json.dumps(json_value(payload), indent=2, allow_nan=False), encoding="utf-8")
    return Path(path)


def load_evidence_bundle(path):
    return verify_evidence_bundle(json.loads(Path(path).read_text(encoding="utf-8")))


def replay_evidence_bundle(path, *, require_versions=True, require_source=True):
    payload = load_evidence_bundle(path)
    return _replay_payload(payload["run"], require_versions=require_versions,
                           require_source=require_source)
