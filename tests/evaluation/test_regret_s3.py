from __future__ import annotations

import pytest

from solverpilot.evaluation import (
    OraclePolicy,
    RegretStatus,
    SelectionDecision,
    build_oracle_table,
    evaluate_selection_decisions,
    summarize_regret_vectors,
)


def row(run_id, backend, objective=10.0, wall=1.0, **kw):
    base = {
        "run_id": run_id, "instance": "p.mps", "instance_sha256": "a"*64,
        "protocol_id": "proto", "environment_id": "env", "backend": backend,
        "state":"solved", "validated":True, "reference_check":"not_checkable",
        "public_status":"valid_optimal", "objective":objective, "wall_s":wall,
        "optimality_evidence":{"independently_verified_optimal":True},
    }
    base.update(kw); return base


def decision(run_id, *, did="d", selector="sel", overhead=0.0, instance="p.mps"):
    return SelectionDecision(did, instance, run_id, selector_id=selector, selector_version="1", selection_overhead_s=overhead)


def test_quality_runtime_and_selection_overhead_are_separate():
    rows = [row("best","a",10,2), row("other","b",12,3)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("other", overhead=0.5)], rows, table)[0]
    assert v.status is RegretStatus.COMPARED
    assert v.quality_regret == 2
    assert v.relative_quality_regret == pytest.approx(0.2)
    assert v.runtime_delta_s == 1
    assert v.selection_overhead_s == 0.5
    assert v.runtime_regret_s == 1.5


def test_maximization_regret_direction():
    rows = [row("best","a",12,2), row("other","b",10,3)]
    table = build_oracle_table(rows, OraclePolicy(objective_sense="maximize"))
    v = evaluate_selection_decisions([decision("other")], rows, table)[0]
    assert v.quality_regret == 2


def test_faster_but_worse_quality_can_have_zero_runtime_regret():
    rows = [row("best","a",10,2), row("other","b",12,1)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("other")], rows, table)[0]
    assert v.quality_regret == 2
    assert v.runtime_delta_s == -1
    assert v.runtime_regret_s == 0


def test_zero_oracle_objective_keeps_absolute_and_blocks_relative():
    rows = [row("best","a",0,1), row("other","b",2,1)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("other")], rows, table)[0]
    assert v.quality_regret == 2
    assert v.relative_quality_regret is None
    assert "zero" in v.reasons[0]


def test_selected_ineligible_failure_not_imputed_as_quality_regret():
    rows = [row("best","a",10), row("bad","b",None, state="solve_error", validated=False, public_status="error")]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("bad")], rows, table)[0]
    assert v.status is RegretStatus.SELECTED_RUN_INELIGIBLE
    assert v.quality_regret is None
    assert v.runtime_regret_s is None
    assert v.failure_regret == 1


def test_hard_timeout_is_explicit_binary_dimension():
    rows = [row("best","a",10), row("bad","b",None, state="hard_timeout", validated=False, public_status=None)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("bad")], rows, table)[0]
    assert v.timeout_regret == 1
    assert v.quality_regret is None


def test_limit_and_infeasible_dimensions_are_separate():
    limit = row("limit","b",9, validated=True, public_status="feasible_limit")
    infeasible = row("inf","c",None, validated=True, public_status="infeasible", reference_check="not_checkable")
    rows = [row("best","a",10), limit, infeasible]
    table = build_oracle_table(rows)
    vectors = evaluate_selection_decisions([
        decision("limit", did="d1", selector="limit"), decision("inf", did="d2", selector="inf")
    ], rows, table)
    by = {v.selector_id:v for v in vectors}
    assert by["limit"].limit_regret == 1
    assert by["inf"].infeasibility_regret == 1
    assert by["limit"].quality_regret is None
    assert by["inf"].quality_regret is None


def test_abstention_and_missing_run_are_not_numeric_regret():
    rows = [row("best","a",10)]
    table = build_oracle_table(rows)
    decisions = [
        SelectionDecision("d1","p.mps",None),
        SelectionDecision("d2","p.mps","does-not-exist"),
    ]
    vectors = evaluate_selection_decisions(decisions, rows, table)
    assert all(v.status is RegretStatus.SELECTED_RUN_MISSING for v in vectors)
    assert all(v.quality_regret is None for v in vectors)


def test_instance_mismatch_fails_closed():
    rows = [row("best","a",10), row("r2","b",11)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("r2", instance="other.mps")], rows, table)[0]
    assert v.status is RegretStatus.INSTANCE_MISMATCH


def test_unavailable_oracle_blocks_comparison():
    rows = [row("bad","a",10, validated=False)]
    table = build_oracle_table(rows)
    v = evaluate_selection_decisions([decision("bad")], rows, table)[0]
    assert v.status is RegretStatus.ORACLE_UNAVAILABLE
    assert not v.comparable


def test_summary_preserves_missingness_and_never_claims_significance():
    rows = [row("best","a",10,1), row("other","b",12,2), row("bad","c",None,state="solve_error",validated=False,public_status="error")]
    table = build_oracle_table(rows)
    vectors = evaluate_selection_decisions([
        decision("best", did="d1"), decision("other", did="d2"), decision("bad", did="d3")
    ], rows, table)
    s = summarize_regret_vectors(vectors)[0]
    assert s.decision_count == 3
    assert s.compared_count == 2
    assert s.incomparable_count == 1
    assert s.quality_regret.observed_count == 2
    assert s.quality_regret.missing_count == 1
    assert s.quality_regret.mean == 1
    assert s.failure_regret_count == 1
    assert s.significance_claim_supported is False


def test_duplicate_decision_and_run_ids_rejected():
    rows = [row("best","a",10)]
    table = build_oracle_table(rows)
    with pytest.raises(ValueError, match="decision_id"):
        evaluate_selection_decisions([decision("best"), decision("best")], rows, table)
    with pytest.raises(ValueError, match="duplicate benchmark"):
        evaluate_selection_decisions([decision("best")], rows + rows, table)
