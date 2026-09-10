from __future__ import annotations

from pathlib import Path
import sqlite3

import numpy as np
import pytest

import solverpilot
from solverpilot import LinearProblem, solve
from solverpilot.history import HistoryDataError, HistoryStore
from solverpilot.intelligence import feature_record_from_problem
from solverpilot.io import SourceProvenance


def make_problem() -> LinearProblem:
    return LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0], variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf], name="s9-lp",
    )


def provenance() -> SourceProvenance:
    return SourceProvenance(
        loader="test", loader_version="1", source_type="json", source_name="case.json",
        source_sha256="1" * 64, source_size_bytes=10, mapping_sha256="2" * 64,
        normalized_payload_sha256="3" * 64, policy_sha256="4" * 64,
        record_count=1, encoding="utf-8",
    )


def test_problem_feature_solve_history_pipeline(tmp_path: Path) -> None:
    problem = make_problem()
    result = solve(problem)
    feature = feature_record_from_problem(problem)
    path = tmp_path / "history.sqlite3"
    with HistoryStore(path) as store:
        rec = store.record_problem_object(problem, provenance=provenance(), metadata={"study": "s9"})
        store.record_feature(feature)
        solve_rec = store.record_solve_result(result, run_id="solve-unit", environment_id="env-test")
        summary = store.summary()
        problem_row = store.problem(problem.data_hash)
        feature_rows = store.feature_rows(problem_data_hash=problem.data_hash)
        solve_rows = store.solve_runs(problem_data_hash=problem.data_hash)
    assert rec.problem_data_hash == problem.data_hash
    assert solve_rec.run_id == "solve-unit"
    assert summary["counts"] == {"problems": 1, "problem_sources": 1, "feature_records": 1, "solve_runs": 1, "benchmark_rows": 0, "artifacts": 0}
    assert problem_row is not None and problem_row["sources"][0]["provenance"]["loader"] == "test"
    assert feature_rows[0]["feature_schema_id"] == feature.feature_schema_id
    assert solve_rows[0]["backend"] == result.trace.backend
    assert solve_rows[0]["validation_valid"] is True


def test_record_solve_result_does_not_persist_x_or_raw_statistics(tmp_path: Path) -> None:
    problem = make_problem()
    result = solve(problem)
    assert result.x is not None
    with HistoryStore(tmp_path / "history.sqlite3") as store:
        store.record_problem_object(problem)
        store.record_solve_result(result, run_id="r")
        row = store.solve_runs()[0]
    encoded = repr(row).lower()
    assert "raw_statistics" not in encoded
    assert "'x'" not in encoded


def test_feature_record_foreign_key_is_fail_closed(tmp_path: Path) -> None:
    feature = feature_record_from_problem(make_problem())
    with HistoryStore(tmp_path / "history.sqlite3") as store:
        with pytest.raises(HistoryDataError):
            store.record_feature(feature)


def test_solver_performance_summary_uses_compact_solve_facts(tmp_path: Path) -> None:
    problem = make_problem()
    result = solve(problem)
    with HistoryStore(tmp_path / "history.sqlite3") as store:
        store.record_problem_object(problem)
        store.record_solve_result(result, run_id="r1")
        store.record_solve_result(result, run_id="r2")
        rows = store.solver_performance_summary()
    assert len(rows) == 1
    assert rows[0]["runs"] == 2
    assert rows[0]["validated_runs"] == 2


def test_top_level_api_remains_frozen() -> None:
    assert len(solverpilot.__all__) == 78
    assert "HistoryStore" not in solverpilot.__all__


def test_same_problem_can_record_multiple_distinct_provenance_sources(tmp_path: Path) -> None:
    problem = make_problem()
    p1 = provenance()
    p2 = SourceProvenance(
        loader="test", loader_version="1", source_type="json", source_name="case-copy.json",
        source_sha256="5" * 64, source_size_bytes=11, mapping_sha256="2" * 64,
        normalized_payload_sha256="3" * 64, policy_sha256="4" * 64,
        record_count=1, encoding="utf-8",
    )
    with HistoryStore(tmp_path / "history.sqlite3") as store:
        store.record_problem_object(problem, provenance=p1)
        store.record_problem_object(problem, provenance=p2)
        row = store.problem(problem.data_hash)
        counts = store.summary()["counts"]
    assert row is not None
    assert len(row["sources"]) == 2
    assert counts["problems"] == 1
    assert counts["problem_sources"] == 2
