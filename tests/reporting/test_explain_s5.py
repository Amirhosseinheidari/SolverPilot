from __future__ import annotations

from dataclasses import replace

import numpy as np
import pytest

import solverpilot
from solverpilot.diagnose import InfeasibilityReport
from solverpilot.inspect import inspect_problem
from solverpilot.plan import CandidatePlan, HealthPolicy, SolveBudget, SolveIntent, SolvePlan
from solverpilot.problem import LinearProblem
from solverpilot.reporting import ClaimDisposition, ClaimKind, explain_result, explain_status
from solverpilot.runtime import SolveResult
from solverpilot.trace import SolveTrace
from solverpilot.validate import PublicStatus, ValidationReport


def _problem() -> LinearProblem:
    return LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[10.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )


def _validation(*, valid: bool = True, warning: str | None = None) -> ValidationReport:
    return ValidationReport(
        valid=valid,
        max_bound_violation=0.0 if valid else 1.0,
        max_constraint_violation=0.0,
        max_integrality_violation=None,
        objective_recomputed=1.0 if valid else None,
        objective_reported=1.0 if valid else None,
        objective_difference=0.0 if valid else None,
        objective_consistent=True if valid else None,
        warnings=() if warning is None else (warning,),
    )


def _result(
    status: PublicStatus,
    *,
    validation: ValidationReport | None = None,
    raw_statistics: dict[str, object] | None = None,
    diagnostics: InfeasibilityReport | None = None,
    plan: SolvePlan | None = None,
    reuse: bool | None = None,
    backend: str = "backend-a",
    trace_warnings: tuple[str, ...] = (),
) -> SolveResult:
    x = np.array([1.0]) if validation is not None else None
    trace = SolveTrace(
        problem_structural_hash="s" * 64,
        problem_data_hash="d" * 64,
        problem_class="lp",
        backend=backend,
        backend_version="1.0",
        reuse_applied=reuse,
        warnings=trace_warnings,
    )
    backend_status = {
        PublicStatus.VALID_OPTIMAL: "optimal",
        PublicStatus.VALID_FEASIBLE: "converged_candidate",
        PublicStatus.FEASIBLE_LIMIT: "limit_feasible",
        PublicStatus.INFEASIBLE: "infeasible",
        PublicStatus.UNBOUNDED: "unbounded",
        PublicStatus.INFEASIBLE_OR_UNBOUNDED: "infeasible_or_unbounded",
        PublicStatus.INVALID_SOLUTION: "optimal",
        PublicStatus.ERROR: "solver_error",
        PublicStatus.UNKNOWN: "unknown",
    }[status]
    return SolveResult(
        status=status,
        x=x,
        objective=1.0 if validation is not None and validation.valid else None,
        backend_status=backend_status,
        validation=validation,
        trace=trace,
        raw_statistics=raw_statistics,
        plan=plan,
        diagnostics=diagnostics,
    )


def _claim(report, kind: ClaimKind):
    return next(c for c in report.claims if c.kind is kind)


def test_every_public_status_has_safe_explanation() -> None:
    for status in PublicStatus:
        text = explain_status(status)
        assert len(text) > 60


def test_valid_optimal_is_qualified_without_complete_independent_evidence() -> None:
    result = _result(PublicStatus.VALID_OPTIMAL, validation=_validation())
    report = explain_result(result)
    claim = _claim(report, ClaimKind.OPTIMALITY)
    assert claim.disposition is ClaimDisposition.QUALIFIED
    assert "did not independently" in claim.rationale.lower()
    assert report.optimality["backend_reported_optimal"] is True
    assert report.optimality["independently_verified_optimal"] is False
    assert "complete independent" in report.summary.lower()


def test_independently_verified_optimality_is_supported() -> None:
    raw = {
        "solverpilot_trust": {
            "backend_reported_optimal": True,
            "primal_validated": True,
            "dual_verified": True,
            "gap_verified": True,
            "certificate_verified": False,
        }
    }
    report = explain_result(_result(PublicStatus.VALID_OPTIMAL, validation=_validation(), raw_statistics=raw))
    claim = _claim(report, ClaimKind.OPTIMALITY)
    assert claim.disposition is ClaimDisposition.SUPPORTED
    assert report.optimality["independently_verified_optimal"] is True


def test_certificate_can_substitute_for_gap_in_independent_optimality_evidence() -> None:
    raw = {
        "solverpilot_trust": {
            "backend_reported_optimal": True,
            "primal_validated": True,
            "dual_verified": True,
            "gap_verified": False,
            "certificate_verified": True,
        }
    }
    report = explain_result(_result(PublicStatus.VALID_OPTIMAL, validation=_validation(), raw_statistics=raw))
    assert _claim(report, ClaimKind.OPTIMALITY).disposition is ClaimDisposition.SUPPORTED


def test_feasible_never_escalates_to_optimal() -> None:
    report = explain_result(_result(PublicStatus.VALID_FEASIBLE, validation=_validation()))
    assert _claim(report, ClaimKind.FEASIBILITY).disposition is ClaimDisposition.SUPPORTED
    assert _claim(report, ClaimKind.OPTIMALITY).disposition is ClaimDisposition.BLOCKED
    assert "does not establish optimality" in report.status_explanation.lower()


def test_feasible_limit_never_escalates_to_optimal() -> None:
    report = explain_result(_result(PublicStatus.FEASIBLE_LIMIT, validation=_validation()))
    assert _claim(report, ClaimKind.FEASIBILITY).disposition is ClaimDisposition.SUPPORTED
    assert _claim(report, ClaimKind.OPTIMALITY).disposition is ClaimDisposition.BLOCKED
    assert "must not be presented as proven optimal" in report.status_explanation.lower()


def test_invalid_candidate_blocks_feasibility_and_optimality() -> None:
    report = explain_result(_result(PublicStatus.INVALID_SOLUTION, validation=_validation(valid=False)))
    assert _claim(report, ClaimKind.FEASIBILITY).disposition is ClaimDisposition.BLOCKED
    assert _claim(report, ClaimKind.OPTIMALITY).disposition is ClaimDisposition.BLOCKED


def test_backend_infeasible_without_diagnostics_is_only_qualified() -> None:
    report = explain_result(_result(PublicStatus.INFEASIBLE))
    claim = _claim(report, ClaimKind.INFEASIBILITY)
    assert claim.disposition is ClaimDisposition.QUALIFIED
    assert "backend reported" in claim.statement.lower()
    assert "no independent" in claim.rationale.lower()


def test_confirmed_diagnostics_support_model_infeasibility_claim() -> None:
    diagnostics = InfeasibilityReport(
        confirmed_infeasible=True,
        static_issues=(),
        iis=None,
        elastic=None,
        conflict=None,
        evidence_order=(),
        warnings=("diagnostic warning",),
    )
    report = explain_result(_result(PublicStatus.INFEASIBLE, diagnostics=diagnostics))
    claim = _claim(report, ClaimKind.INFEASIBILITY)
    assert claim.disposition is ClaimDisposition.SUPPORTED
    assert "supplied mathematical model" in claim.qualifiers[0]
    assert report.diagnostics is not None
    assert report.diagnostics["confirmed_infeasible"] is True


def test_non_infeasible_status_blocks_infeasibility_claim() -> None:
    report = explain_result(_result(PublicStatus.ERROR))
    assert _claim(report, ClaimKind.INFEASIBILITY).disposition is ClaimDisposition.BLOCKED


def test_unbounded_backend_status_is_qualified_not_independent_proof() -> None:
    report = explain_result(_result(PublicStatus.UNBOUNDED))
    claim = _claim(report, ClaimKind.UNBOUNDEDNESS)
    assert claim.disposition is ClaimDisposition.QUALIFIED
    assert "independently proven" in claim.qualifiers[0].lower()


def test_infeasible_or_unbounded_supports_neither_specific_claim() -> None:
    report = explain_result(_result(PublicStatus.INFEASIBLE_OR_UNBOUNDED))
    assert _claim(report, ClaimKind.INFEASIBILITY).disposition is ClaimDisposition.BLOCKED
    assert _claim(report, ClaimKind.UNBOUNDEDNESS).disposition is ClaimDisposition.BLOCKED
    assert "neither conclusion" in report.status_explanation.lower()


def test_planner_selection_is_factual_not_superiority_claim() -> None:
    fp = inspect_problem(_problem())
    plan = SolvePlan(
        selected_backend="backend-a",
        candidates=(CandidatePlan("backend-a", 1.0, ("compatible",)),),
        intent=SolveIntent.BALANCED,
        budget=SolveBudget(),
        fingerprint=fp,
        strategy="capability",
        rationale=("compatible",),
        evidence_level="capability_only",
        health_policy=HealthPolicy.IGNORE,
    )
    report = explain_result(_result(PublicStatus.VALID_FEASIBLE, validation=_validation(), plan=plan))
    claim = _claim(report, ClaimKind.PLANNER_SELECTION)
    assert claim.disposition is ClaimDisposition.SUPPORTED
    assert "globally fastest" in claim.qualifiers[0]
    assert report.planner is not None
    assert report.planner["selected_backend"] == "backend-a"


def test_reuse_is_not_escalated_to_performance_improvement() -> None:
    report = explain_result(_result(PublicStatus.VALID_FEASIBLE, validation=_validation(), reuse=True))
    claim = _claim(report, ClaimKind.REOPTIMIZATION)
    assert claim.disposition is ClaimDisposition.SUPPORTED
    assert "does not" in claim.qualifiers[0].lower()
    assert "performance" in claim.qualifiers[0].lower()


def test_no_reuse_claim_when_trace_does_not_record_reuse() -> None:
    report = explain_result(_result(PublicStatus.VALID_FEASIBLE, validation=_validation(), reuse=None))
    assert all(c.kind is not ClaimKind.REOPTIMIZATION for c in report.claims)


def test_validation_payload_is_structured_and_preserves_warnings() -> None:
    report = explain_result(_result(PublicStatus.VALID_FEASIBLE, validation=_validation(warning="scaled tolerance")))
    assert report.validation is not None
    assert report.validation["valid"] is True
    assert report.validation["warnings"] == ("scaled tolerance",)


def test_warnings_are_deduplicated_across_validation_trace_and_diagnostics() -> None:
    diagnostics = InfeasibilityReport(
        confirmed_infeasible=True,
        static_issues=(),
        iis=None,
        elastic=None,
        conflict=None,
        evidence_order=(),
        warnings=("same", "diagnostic"),
    )
    result = _result(
        PublicStatus.INFEASIBLE,
        validation=_validation(warning="same"),
        diagnostics=diagnostics,
        trace_warnings=("same", "trace"),
    )
    report = explain_result(result)
    assert report.warnings == ("same", "trace", "diagnostic")


def test_explain_result_does_not_mutate_source_result() -> None:
    result = _result(PublicStatus.VALID_FEASIBLE, validation=_validation())
    before = result.raw_statistics
    _ = explain_result(result)
    assert result.raw_statistics is before
    assert result.status is PublicStatus.VALID_FEASIBLE


def test_explain_result_requires_solve_result() -> None:
    with pytest.raises(TypeError, match="SolveResult"):
        explain_result(object())  # type: ignore[arg-type]


def test_top_level_api_remains_frozen() -> None:
    assert len(solverpilot.__all__) == 78
    assert "explain_result" not in solverpilot.__all__
    assert not hasattr(solverpilot, "ExplanationReport")
