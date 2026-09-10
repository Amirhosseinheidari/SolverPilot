from __future__ import annotations

import json

import pytest

from solverpilot.reporting import (
    ClaimDisposition,
    ClaimKind,
    ExplanationClaim,
    ExplanationReport,
    render_json,
    render_markdown,
)


def _claim(*, statement: str = "Recorded fact") -> ExplanationClaim:
    return ExplanationClaim(
        claim_id="claim.fact",
        kind=ClaimKind.FEASIBILITY,
        disposition=ClaimDisposition.SUPPORTED,
        statement=statement,
        rationale="Structured evidence exists.",
        evidence_refs=("result.validation",),
    )


def _report(**overrides):
    values = dict(
        status="valid_feasible",
        summary="A validated candidate exists.",
        status_explanation="Feasibility is not optimality.",
        problem_class="lp",
        backend="backend-a",
        backend_version="1.0",
        objective=1.0,
        validation={"valid": True},
        optimality={
            "backend_reported_optimal": False,
            "primal_validated": True,
            "dual_verified": False,
            "gap_verified": False,
            "certificate_verified": False,
            "independently_verified_optimal": False,
        },
        planner=None,
        runtime={"reuse_applied": False, "reuse_mode": None, "total_s": 0.0, "solve_s": 0.0, "validate_s": 0.0, "diagnose_s": 0.0},
        diagnostics=None,
        claims=(_claim(),),
        warnings=(),
    )
    values.update(overrides)
    return ExplanationReport(**values)


def test_model_serializes_enums_as_strings() -> None:
    payload = _report().to_dict()
    assert payload["claims"][0]["kind"] == "feasibility"
    assert payload["claims"][0]["disposition"] == "supported"


def test_json_renderer_is_deterministic_and_valid() -> None:
    report = _report()
    first = render_json(report)
    second = render_json(report)
    assert first == second
    assert json.loads(first)["schema_version"] == "0.1"


def test_nonfinite_values_are_rejected_at_report_boundary() -> None:
    with pytest.raises(ValueError, match="NaN or infinity"):
        _report(objective=float("nan"))
    with pytest.raises(ValueError, match="NaN or infinity"):
        _report(validation={"valid": True, "value": float("inf")})


def test_non_json_safe_nested_values_are_rejected() -> None:
    with pytest.raises(ValueError, match="not JSON-safe"):
        _report(validation={"bad": {"set"}})


def test_duplicate_claim_ids_are_rejected() -> None:
    claim = _claim()
    with pytest.raises(ValueError, match="unique"):
        _report(claims=(claim, claim))


def test_duplicate_evidence_refs_are_rejected() -> None:
    with pytest.raises(ValueError, match="duplicates"):
        ExplanationClaim(
            claim_id="x",
            kind=ClaimKind.FEASIBILITY,
            disposition=ClaimDisposition.SUPPORTED,
            statement="x",
            rationale="x",
            evidence_refs=("a", "a"),
        )


def test_empty_claim_fields_are_rejected() -> None:
    with pytest.raises(ValueError, match="claim_id"):
        ExplanationClaim("", ClaimKind.FEASIBILITY, ClaimDisposition.SUPPORTED, "x", "x")
    with pytest.raises(ValueError, match="statement"):
        ExplanationClaim("x", ClaimKind.FEASIBILITY, ClaimDisposition.SUPPORTED, "", "x")
    with pytest.raises(ValueError, match="rationale"):
        ExplanationClaim("x", ClaimKind.FEASIBILITY, ClaimDisposition.SUPPORTED, "x", "")


def test_markdown_escapes_raw_html_table_pipes_and_newlines() -> None:
    report = _report(
        backend="<script>alert(1)</script>|backend",
        summary="## APPROVED\n<script>alert(1)</script>",
        claims=(_claim(statement="value | injected\n## heading"),),
        warnings=("<b>warning</b>\n# heading",),
    )
    markdown = render_markdown(report)
    assert "<script>" not in markdown
    assert "&lt;script&gt;" in markdown
    assert "\\|" in markdown
    assert "\n## APPROVED\n" not in markdown
    assert "\n# heading\n" not in markdown


def test_renderers_require_explanation_report() -> None:
    with pytest.raises(TypeError, match="ExplanationReport"):
        render_json(object())  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="ExplanationReport"):
        render_markdown(object())  # type: ignore[arg-type]


def test_manual_supported_optimality_cannot_bypass_structured_evidence() -> None:
    claim = ExplanationClaim(
        "claim.opt", ClaimKind.OPTIMALITY, ClaimDisposition.SUPPORTED, "This is optimal.", "Caller asserted it.",
        evidence_refs=("result.optimality_evidence",),
    )
    with pytest.raises(ValueError, match="not authorized"):
        _report(claims=(claim,))


def test_unknown_evidence_reference_is_rejected() -> None:
    claim = ExplanationClaim(
        "claim.unknown", ClaimKind.FEASIBILITY, ClaimDisposition.QUALIFIED, "Maybe feasible.", "Unknown evidence.",
        evidence_refs=("missing.reference",),
    )
    with pytest.raises(ValueError, match="unknown evidence"):
        _report(claims=(claim,))


def test_supported_unboundedness_is_rejected_without_independent_proof_field() -> None:
    claim = ExplanationClaim(
        "claim.unbounded", ClaimKind.UNBOUNDEDNESS, ClaimDisposition.SUPPORTED, "Proven unbounded.", "Caller asserted it.",
        evidence_refs=("result.backend_status",),
    )
    with pytest.raises(ValueError, match="not authorized"):
        _report(claims=(claim,))


def test_report_claim_safe_is_true_for_explicit_qualified_and_blocked_claims() -> None:
    claims = (
        ExplanationClaim(
            "a", ClaimKind.OPTIMALITY, ClaimDisposition.QUALIFIED, "Backend reported optimality.", "Incomplete proof."
        ),
        ExplanationClaim("b", ClaimKind.UNBOUNDEDNESS, ClaimDisposition.BLOCKED, "Not established.", "No evidence."),
    )
    assert _report(claims=claims).claim_safe is True
