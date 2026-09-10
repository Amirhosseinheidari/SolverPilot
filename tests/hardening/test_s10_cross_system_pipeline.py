from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solverpilot import LinearProblem, solve
from solverpilot.evaluation import SelectionDecision, build_oracle_table, evaluate_selection_decisions
from solverpilot.history import HistoryStore
from solverpilot.intelligence import feature_record_from_problem
from solverpilot.io import load_json
from solverpilot.reporting import explain_result, render_json


def test_io_provenance_feature_solve_report_history_oracle_regret_pipeline(tmp_path: Path) -> None:
    source = tmp_path / "problem.json"
    source.write_text(
        json.dumps(
            {
                "A": [[1.0, 1.0]],
                "c": [1.0, 2.0],
                "variable_lower": [0.0, 0.0],
                "variable_upper": [1.0, 1.0],
                "constraint_lower": [1.0],
                "constraint_upper": [1.0],
            }
        ),
        encoding="utf-8",
    )
    loaded = load_json(source)
    payload = loaded.mutable_payload()
    problem = LinearProblem.from_data(
        A=payload["A"],
        c=payload["c"],
        variable_lower=payload["variable_lower"],
        variable_upper=payload["variable_upper"],
        constraint_lower=payload["constraint_lower"],
        constraint_upper=payload["constraint_upper"],
        name="s10-cross-system",
        metadata=loaded.provenance.as_problem_metadata(),
    )
    feature = feature_record_from_problem(problem, instance_id="case-1")
    result = solve(problem)
    assert result.validation is not None and result.validation.valid
    report = explain_result(result)
    rendered = render_json(report)
    assert '"claim' in rendered
    assert "raw_statistics" not in rendered

    db = tmp_path / "history.sqlite3"
    with HistoryStore(db) as store:
        store.record_problem_object(problem, provenance=loaded.provenance, metadata={"pipeline": "s10"})
        store.record_feature(feature)
        store.record_solve_result(result, run_id="run-a", environment_id="env-s10")
        common = {
            "protocol_id": "proto-s10",
            "environment_id": "env-s10",
            "instance": "case-1",
            "instance_sha256": loaded.provenance.source_sha256,
            "problem_data_hash": problem.data_hash,
            "repetition": 0,
            "state": "solved",
            "public_status": result.status.value,
            "validated": True,
            "reference_check": "objective_matches_optimum",
            "objective": float(result.objective),
        }
        store.record_benchmark_row({**common, "run_id": "bench-a", "backend": "a", "wall_s": 1.0})
        store.record_benchmark_row({**common, "run_id": "bench-b", "backend": "b", "wall_s": 2.0})
        rows = store.benchmark_rows(protocol_id="proto-s10")
        stored_problem = store.problem(problem.data_hash)
        stored_solves = store.solve_runs(problem_data_hash=problem.data_hash)

    assert stored_problem is not None
    assert stored_problem["sources"][0]["provenance"]["source_sha256"] == loaded.provenance.source_sha256
    assert stored_solves[0]["run_id"] == "run-a"
    assert "x" not in stored_solves[0]

    oracle = build_oracle_table(rows)
    assert oracle.entries[0].oracle_run_ids == ("bench-a", "bench-b")
    vectors = evaluate_selection_decisions(
        [SelectionDecision("decision-1", "case-1", "bench-b", selection_overhead_s=0.25)],
        rows,
        oracle,
    )
    assert len(vectors) == 1 and vectors[0].comparable
    assert vectors[0].quality_regret == 0.0
    assert np.isclose(vectors[0].runtime_regret_s, 1.25)
