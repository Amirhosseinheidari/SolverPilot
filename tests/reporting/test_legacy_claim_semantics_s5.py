from __future__ import annotations

from solverpilot.reporting import explain_status
from solverpilot.validate import PublicStatus


def test_legacy_feasibility_safety_semantic_is_preserved() -> None:
    text = explain_status(PublicStatus.VALID_FEASIBLE).lower()
    assert "feasibility" in text
    assert "does not establish optimality" in text


def test_legacy_infeasibility_caution_semantic_is_preserved() -> None:
    text = explain_status(PublicStatus.INFEASIBLE).lower()
    assert "backend reported" in text
    assert "not an independently" in text


def test_error_status_does_not_authorize_normal_decision_claim() -> None:
    text = explain_status(PublicStatus.ERROR).lower()
    assert "no feasibility" in text
    assert "optimality" in text
