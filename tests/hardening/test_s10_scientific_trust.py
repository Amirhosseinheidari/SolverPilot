from __future__ import annotations

import pytest

from solverpilot.applications.routing import VRPRoute, VRPSolution
from solverpilot.applications.tsp import TSPSolution
from solverpilot.evaluation import OracleStatus, build_oracle_table
from solverpilot.reporting import (
    ClaimDisposition,
    ClaimKind,
    ExplanationClaim,
    ExplanationReport,
    render_json,
    render_markdown,
)


def _oracle_row(run_id: str = "r1", **patch):
    row = {
        "run_id": run_id,
        "instance": "p.mps",
        "instance_sha256": "a" * 64,
        "protocol_id": "proto",
        "environment_id": "env",
        "backend": "scipy-highs-bridge",
        "state": "solved",
        "validated": True,
        "reference_check": "objective_matches_optimum",
        "public_status": "valid_optimal",
        "objective": 10.0,
        "wall_s": 1.0,
    }
    row.update(patch)
    return row


def test_best_known_reference_is_not_treated_as_exact_oracle_evidence():
    entry = build_oracle_table([
        _oracle_row(reference_check="meets_or_beats_best_known", public_status="valid_feasible")
    ]).entries[0]
    assert entry.status is OracleStatus.NO_ELIGIBLE_CANDIDATE
    assert entry.oracle_run_ids == ()


@pytest.mark.parametrize(
    ("field", "value", "status"),
    [
        ("protocol_id", None, OracleStatus.PROTOCOL_MISMATCH),
        ("environment_id", "", OracleStatus.ENVIRONMENT_MISMATCH),
        ("instance_sha256", None, OracleStatus.INSTANCE_IDENTITY_MISMATCH),
        ("instance_sha256", "not-a-sha", OracleStatus.INSTANCE_IDENTITY_MISMATCH),
    ],
)
def test_required_oracle_comparability_metadata_must_be_present_and_valid(field, value, status):
    row = _oracle_row()
    row[field] = value
    entry = build_oracle_table([row]).entries[0]
    assert entry.status is status
    assert entry.oracle_run_ids == ()


def _report_inputs():
    return dict(
        status="valid_optimal",
        summary="candidate",
        status_explanation="qualified",
        problem_class="lp",
        backend="b",
        backend_version="1",
        objective=1.0,
        validation={"valid": True},
        optimality={"independently_verified_optimal": True},
        planner=None,
        runtime={"reuse_applied": False},
        diagnostics=None,
        claims=(
            ExplanationClaim(
                "optimal",
                ClaimKind.OPTIMALITY,
                ClaimDisposition.SUPPORTED,
                "independently proven",
                "structured evidence authorizes this claim",
                ("result.optimality_evidence",),
            ),
        ),
    )


def test_report_structured_evidence_is_deeply_immutable_after_validation():
    source = _report_inputs()
    report = ExplanationReport(**source)
    # Mutating caller-owned dictionaries must not alter the report snapshot.
    source["optimality"]["independently_verified_optimal"] = False
    source["validation"]["valid"] = False
    assert report.optimality["independently_verified_optimal"] is True
    assert report.validation is not None and report.validation["valid"] is True
    with pytest.raises(TypeError):
        report.optimality["independently_verified_optimal"] = False  # type: ignore[index]
    assert report.claim_safe
    assert "supported" in render_json(report)
    assert "supported" in render_markdown(report)


def test_report_claim_enums_are_normalized_so_string_types_cannot_bypass_claim_safety():
    claim = ExplanationClaim(
        "optimal",
        "optimality",  # type: ignore[arg-type]
        "supported",  # type: ignore[arg-type]
        "claim",
        "reason",
        ("result.optimality_evidence",),
    )
    assert claim.kind is ClaimKind.OPTIMALITY
    assert claim.disposition is ClaimDisposition.SUPPORTED
    with pytest.raises(ValueError, match="not authorized"):
        ExplanationReport(
            status="valid_optimal",
            summary="candidate",
            status_explanation="qualified",
            problem_class="lp",
            backend="b",
            backend_version="1",
            objective=1.0,
            validation={"valid": True},
            optimality={"independently_verified_optimal": False},
            planner=None,
            runtime={"reuse_applied": False},
            diagnostics=None,
            claims=(claim,),
        )


def test_application_solution_constructors_cannot_self_issue_independent_proof():
    with pytest.raises(Exception, match="only be issued"):
        TSPSolution((0, 1, 0), 2.0, method="manual", is_exact=True, optimality_proven=True)
    with pytest.raises(Exception, match="only be issued"):
        VRPSolution(
            (VRPRoute("v", ("a",)),),
            2.0,
            method="manual",
            is_exact=True,
            optimality_proven=True,
        )

@pytest.mark.parametrize(
    "field",
    [
        "runtime_tie_breaker",
        "require_same_protocol",
        "require_same_environment",
        "require_same_instance_sha256",
        "include_auto_backend",
    ],
)
def test_oracle_policy_boolean_fields_reject_truthy_strings(field):
    from solverpilot.evaluation import OraclePolicy

    with pytest.raises(ValueError, match="boolean"):
        OraclePolicy(**{field: "false"})


def test_claim_sequence_fields_reject_plain_strings():
    with pytest.raises(ValueError, match="sequence of strings"):
        ExplanationClaim(
            "c",
            ClaimKind.FEASIBILITY,
            ClaimDisposition.QUALIFIED,
            "statement",
            "rationale",
            "result.validation",  # type: ignore[arg-type]
        )
