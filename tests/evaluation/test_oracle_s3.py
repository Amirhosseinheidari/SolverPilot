from __future__ import annotations

import pytest

from solverpilot.evaluation import (
    CandidateExclusionReason,
    ObjectiveSense,
    OraclePolicy,
    OracleStatus,
    build_oracle_table,
)


def row(run_id, backend, objective=10.0, wall=1.0, **kw):
    base = {
        "run_id": run_id,
        "instance": "p.mps",
        "instance_sha256": "a" * 64,
        "protocol_id": "proto",
        "environment_id": "env",
        "backend": backend,
        "state": "solved",
        "validated": True,
        "reference_check": "not_checkable",
        "public_status": "valid_optimal",
        "optimality_evidence": {"independently_verified_optimal": True},
        "objective": objective,
        "wall_s": wall,
    }
    base.update(kw)
    return base


def test_minimize_and_maximize_are_direction_aware():
    rows = [row("r1", "a", 10), row("r2", "b", 12)]
    e = build_oracle_table(rows).entries[0]
    assert e.status is OracleStatus.AVAILABLE
    assert e.oracle_objective == 10
    assert e.oracle_run_ids == ("r1",)
    e = build_oracle_table(rows, OraclePolicy(objective_sense="maximize")).entries[0]
    assert e.objective_sense is ObjectiveSense.MAXIMIZE
    assert e.oracle_objective == 12
    assert e.oracle_run_ids == ("r2",)


def test_tied_objectives_are_preserved_and_runtime_tiebreak_is_explicit():
    rows = [row("r1", "a", 10, 2), row("r2", "b", 10, 1)]
    e = build_oracle_table(rows).entries[0]
    assert e.oracle_run_ids == ("r1", "r2")
    assert e.fastest_oracle_runtime_s == 1
    e = build_oracle_table(rows, OraclePolicy(runtime_tie_breaker=True)).entries[0]
    assert e.oracle_run_ids == ("r2",)


@pytest.mark.parametrize("patch,reason", [
    ({"state":"solve_error"}, CandidateExclusionReason.NOT_SOLVED),
    ({"validated":False}, CandidateExclusionReason.NOT_VALIDATED),
    ({"reference_check":"not_checkable","public_status":"valid_feasible"}, CandidateExclusionReason.NOT_EXACT_SUCCESS),
    ({"objective":None}, CandidateExclusionReason.MISSING_OBJECTIVE),
    ({"objective":float("inf")}, CandidateExclusionReason.NONFINITE_OBJECTIVE),
    ({"wall_s":None}, CandidateExclusionReason.MISSING_RUNTIME),
    ({"wall_s":float("nan")}, CandidateExclusionReason.NONFINITE_RUNTIME),
    ({"wall_s":-1.0}, CandidateExclusionReason.NEGATIVE_RUNTIME),
])
def test_candidate_exclusions_are_explicit(patch, reason):
    good = row("good", "a", 10)
    bad = row("bad", "b", **patch)
    entry = build_oracle_table([good, bad]).entries[0]
    exclusion = next(x for x in entry.exclusions if x.run_id == "bad")
    assert exclusion.reason is reason
    assert entry.oracle_run_ids == ("good",)


def test_auto_rows_are_not_oracle_candidates_by_default():
    entry = build_oracle_table([row("r1","a",10), row("auto","@auto",1)]).entries[0]
    assert entry.oracle_run_ids == ("r1",)
    assert any(x.reason is CandidateExclusionReason.AUTO_BACKEND for x in entry.exclusions)


def test_reference_success_can_be_oracle_candidate():
    entry = build_oracle_table([
        row("r1","a",10, reference_check="objective_matches_optimum", public_status="valid_feasible")
    ]).entries[0]
    assert entry.status is OracleStatus.AVAILABLE


@pytest.mark.parametrize("field,value,status", [
    ("protocol_id", "proto2", OracleStatus.PROTOCOL_MISMATCH),
    ("environment_id", "env2", OracleStatus.ENVIRONMENT_MISMATCH),
    ("instance_sha256", "b"*64, OracleStatus.INSTANCE_IDENTITY_MISMATCH),
])
def test_comparability_mismatch_fails_closed(field, value, status):
    other = row("r2","b",11)
    other[field] = value
    entry = build_oracle_table([row("r1","a",10), other]).entries[0]
    assert entry.status is status
    assert not entry.oracle_run_ids


def test_no_eligible_candidate_is_explicit():
    entry = build_oracle_table([row("r1","a",10, validated=False)]).entries[0]
    assert entry.status is OracleStatus.NO_ELIGIBLE_CANDIDATE


def test_nested_runtime_field_supported():
    r = row("r1","a",10)
    r["trace"] = {"solve_s": 0.25}
    entry = build_oracle_table([r], OraclePolicy(runtime_field="trace.solve_s")).entries[0]
    assert entry.fastest_oracle_runtime_s == 0.25


def test_duplicate_run_ids_rejected():
    with pytest.raises(ValueError, match="unique"):
        build_oracle_table([row("r","a"), row("r","b")])
