"""Portable core LP/QP evidence envelopes; recorded claims are not re-certified on load."""

import json
from pathlib import Path

from .manifest import (
    _replay_payload, _seal, _summary_payload, _verify_integrity, json_value, run_manifest,
)
from .result import SolveResult


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


def verify_evidence_bundle(payload):
    """Check envelope integrity and linkage, never trust a JSON proof claim."""
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
    if payload["summary"] != run["summary"]:
        raise ValueError("evidence summaries disagree")
    if any(run[key] != payload["summary"][key] for key in ("status", "backend")):
        raise ValueError("evidence run disagrees with summary")
    if payload["explanation"]["runtime"]["source_sha256"] != run["code_identity"]["source_sha256"]:
        raise ValueError("evidence source identity mismatch")
    if any(payload["explanation"][key] != payload["summary"][key]
           for key in ("status", "objective", "backend")):
        raise ValueError("evidence explanation disagrees with summary")
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
