from __future__ import annotations

import json
import math
from pathlib import Path
import sqlite3
from concurrent.futures import ThreadPoolExecutor

import pytest

from solverpilot.history import (
    HISTORY_SCHEMA_VERSION,
    HistoryDataError,
    HistorySchemaError,
    HistoryStore,
    HistoryStoreError,
    ProblemHistoryRecord,
    SolveHistoryRecord,
)

A = "a" * 64
B = "b" * 64


def problem_record() -> ProblemHistoryRecord:
    return ProblemHistoryRecord(A, B, "lp", 2, 1, metadata={"safe": True})


def solve_record(i: int) -> SolveHistoryRecord:
    return SolveHistoryRecord(
        run_id=f"run-{i}",
        problem_data_hash=A,
        problem_structural_hash=B,
        problem_class="lp",
        backend="scipy-highs",
        backend_version="x",
        public_status="valid_optimal",
        backend_status="optimal",
        objective=float(i),
        validation_valid=True,
        independently_verified_optimal=False,
        total_s=0.01 + i / 1000,
    )


def test_initialize_empty_db_and_reopen_idempotently(tmp_path: Path) -> None:
    path = tmp_path / "history.sqlite3"
    with HistoryStore(path) as store:
        summary = store.summary()
        assert summary["schema_version"] == HISTORY_SCHEMA_VERSION
        assert summary["raw_problem_payloads_persisted"] is False
        assert summary["raw_solution_vectors_persisted"] is False
        store.record_problem(problem_record())
    with HistoryStore(path) as store:
        assert store.summary()["counts"]["problems"] == 1
    con = sqlite3.connect(path)
    assert con.execute("PRAGMA user_version").fetchone()[0] == HISTORY_SCHEMA_VERSION
    con.close()


def test_newer_schema_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "future.sqlite3"
    con = sqlite3.connect(path)
    con.execute("PRAGMA user_version = 999")
    con.commit(); con.close()
    with pytest.raises(HistorySchemaError):
        HistoryStore(path)


def test_current_version_with_missing_identity_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "alien.sqlite3"
    con = sqlite3.connect(path)
    con.execute("PRAGMA user_version = 1")
    con.commit(); con.close()
    with pytest.raises((HistorySchemaError, sqlite3.OperationalError)):
        HistoryStore(path)


def test_directory_and_symlink_paths_rejected(tmp_path: Path) -> None:
    with pytest.raises(HistoryStoreError):
        HistoryStore(tmp_path)
    target = tmp_path / "target.sqlite3"
    target.touch()
    link = tmp_path / "link.sqlite3"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation unavailable")
    with pytest.raises(HistoryStoreError):
        HistoryStore(link)


def test_transaction_rolls_back(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        with pytest.raises(RuntimeError):
            with store.transaction() as con:
                con.execute(
                    "INSERT INTO problems(problem_data_hash,problem_structural_hash,problem_class,n_variables,n_constraints,metadata_json,created_at_utc) VALUES(?,?,?,?,?,?,?)",
                    (A, B, "lp", 1, 1, "{}", "now"),
                )
                raise RuntimeError("abort")
        assert store.summary()["counts"]["problems"] == 0


def test_problem_upsert_is_idempotent(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_problem(problem_record())
        store.record_problem(problem_record())
        assert store.summary()["counts"]["problems"] == 1


def test_solve_requires_known_problem(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        with pytest.raises(HistoryDataError):
            store.record_solve(solve_record(1))
        assert store.summary()["counts"]["solve_runs"] == 0


def test_concurrent_writes_are_serialized_safely(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_problem(problem_record())
        with ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(lambda i: store.record_solve(solve_record(i)), range(30)))
        rows = store.solve_runs(problem_data_hash=A)
        assert len(rows) == 30
        assert {row["run_id"] for row in rows} == {f"run-{i}" for i in range(30)}


def test_benchmark_row_is_whitelisted_and_strips_opaque_payload(tmp_path: Path) -> None:
    row = {
        "schema_version": "1.3", "protocol_id": "p", "environment_id": "e", "run_id": "r",
        "instance": "case", "instance_sha256": A, "backend": "b", "repetition": 0,
        "state": "solved", "public_status": "valid_optimal", "validated": True,
        "objective": 2.0, "wall_s": 0.1, "reference_check": "not_checkable",
        "trace": {"sensitive": "trace"}, "x": [1, 2], "solution": {"raw": True},
        "arbitrary_raw_payload": {"secret": "should-not-persist"},
    }
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_benchmark_row(row)
        saved = store.benchmark_rows()[0]
    assert saved["run_id"] == "r"
    assert "trace" not in saved and "x" not in saved and "solution" not in saved
    assert "arbitrary_raw_payload" not in saved


@pytest.mark.parametrize("field,value", [("objective", math.nan), ("wall_s", math.inf), ("repetition", -1), ("validated", 1)])
def test_benchmark_row_rejects_invalid_values(tmp_path: Path, field: str, value) -> None:
    row = {"run_id": "r", "protocol_id": "p", "instance": "i", "backend": "b", "repetition": 0, "state": "solved"}
    row[field] = value
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        with pytest.raises(HistoryDataError):
            store.record_benchmark_row(row)


def test_artifact_metadata_validation(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_artifact(artifact_id="a", artifact_type="json", size_bytes=3, sha256="c" * 64, metadata={"ok": True})
        assert store.summary()["counts"]["artifacts"] == 1
        with pytest.raises(HistoryDataError):
            store.record_artifact(artifact_id="bad", artifact_type="json", metadata={"x": math.nan})


def test_database_does_not_define_raw_solution_or_problem_payload_columns(tmp_path: Path) -> None:
    path = tmp_path / "h.sqlite3"
    with HistoryStore(path):
        pass
    con = sqlite3.connect(path)
    schema = "\n".join(row[0] for row in con.execute("SELECT sql FROM sqlite_master WHERE type='table' AND sql IS NOT NULL"))
    con.close()
    lowered = schema.lower()
    assert " raw_problem" not in lowered
    assert " solution_vector" not in lowered
    assert " x blob" not in lowered


def test_same_problem_hash_with_conflicting_identity_is_rejected(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_problem(problem_record())
        conflicting = ProblemHistoryRecord(A, "c" * 64, "lp", 2, 1, metadata={"safe": True})
        with pytest.raises(HistoryDataError):
            store.record_problem(conflicting)
        assert store.summary()["counts"]["problems"] == 1


def test_same_run_id_with_conflicting_content_is_rejected(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_problem(problem_record())
        store.record_solve(solve_record(1))
        changed = SolveHistoryRecord(
            run_id="run-1", problem_data_hash=A, problem_structural_hash=B, problem_class="lp",
            backend="scipy-highs", backend_version="x", public_status="valid_optimal",
            backend_status="optimal", objective=999.0, validation_valid=True,
            independently_verified_optimal=False, total_s=0.011,
        )
        with pytest.raises(HistoryDataError):
            store.record_solve(changed)
        assert store.solve_runs()[0]["objective"] == 1.0


def test_same_benchmark_run_id_with_conflicting_content_is_rejected(tmp_path: Path) -> None:
    base = {"run_id": "r", "protocol_id": "p", "instance": "i", "backend": "b", "repetition": 0, "state": "solved", "objective": 1.0}
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_benchmark_row(base)
        with pytest.raises(HistoryDataError):
            store.record_benchmark_row({**base, "objective": 2.0})
        assert store.benchmark_rows()[0]["objective"] == 1.0


def test_integrity_check_reports_clean_database(tmp_path: Path) -> None:
    with HistoryStore(tmp_path / "h.sqlite3") as store:
        store.record_problem(problem_record())
        report = store.integrity_check()
    assert report == {"ok": True, "integrity": ["ok"], "foreign_key_violations": []}


def test_corrupt_non_sqlite_file_raises_history_error(tmp_path: Path) -> None:
    path = tmp_path / "corrupt.sqlite3"
    path.write_bytes(b"this is not sqlite")
    with pytest.raises(HistoryStoreError, match="valid/compatible SQLite"):
        HistoryStore(path)
