from __future__ import annotations

import math
import pytest

from solverpilot.history import HistoryDataError, ProblemHistoryRecord, SolveHistoryRecord

SHA_A = "a" * 64
SHA_B = "b" * 64


def test_problem_record_roundtrip_contract() -> None:
    record = ProblemHistoryRecord(
        problem_data_hash=SHA_A,
        problem_structural_hash=SHA_B,
        problem_class="lp",
        n_variables=2,
        n_constraints=1,
        source_provenance={"loader": "test", "nested": [1, True, None]},
        metadata={"tag": "unit"},
    )
    payload = record.to_dict()
    assert payload["problem_data_hash"] == SHA_A
    assert payload["source_provenance"]["nested"] == [1, True, None]


@pytest.mark.parametrize("value", ["", "x", "g" * 64, "a" * 63])
def test_problem_record_rejects_invalid_hash(value: str) -> None:
    with pytest.raises(HistoryDataError):
        ProblemHistoryRecord(value, SHA_B, "lp", 1, 1)


@pytest.mark.parametrize("value", [-1, True, 1.5])
def test_problem_record_rejects_invalid_dimensions(value) -> None:
    with pytest.raises(HistoryDataError):
        ProblemHistoryRecord(SHA_A, SHA_B, "lp", value, 1)


def test_problem_record_rejects_nonfinite_metadata() -> None:
    with pytest.raises(HistoryDataError):
        ProblemHistoryRecord(SHA_A, SHA_B, "lp", 1, 1, metadata={"bad": math.nan})


def test_solve_record_is_compact_and_validated() -> None:
    record = SolveHistoryRecord(
        run_id="run-1",
        problem_data_hash=SHA_A,
        problem_structural_hash=SHA_B,
        problem_class="lp",
        backend="scipy-highs",
        backend_version="1",
        public_status="valid_optimal",
        backend_status="optimal",
        objective=1.25,
        validation_valid=True,
        independently_verified_optimal=False,
        total_s=0.2,
        environment_id="env",
        metadata={"experiment": "s9"},
    )
    payload = record.to_dict()
    assert payload["objective"] == 1.25
    assert "x" not in payload
    assert "raw_statistics" not in payload


@pytest.mark.parametrize("field,value", [("objective", math.inf), ("objective", math.nan), ("total_s", -1.0), ("total_s", math.inf)])
def test_solve_record_rejects_invalid_numbers(field: str, value: float) -> None:
    kwargs = dict(
        run_id="run",
        problem_data_hash=SHA_A,
        problem_structural_hash=SHA_B,
        problem_class="lp",
        backend=None,
        backend_version=None,
        public_status="unknown",
        backend_status="unknown",
        objective=None,
        validation_valid=None,
        independently_verified_optimal=False,
        total_s=None,
    )
    kwargs[field] = value
    with pytest.raises(HistoryDataError):
        SolveHistoryRecord(**kwargs)


def test_solve_record_requires_real_bools() -> None:
    with pytest.raises(HistoryDataError):
        SolveHistoryRecord("r", SHA_A, SHA_B, "lp", None, None, "unknown", "unknown", None, "yes", False, None)
    with pytest.raises(HistoryDataError):
        SolveHistoryRecord("r", SHA_A, SHA_B, "lp", None, None, "unknown", "unknown", None, None, 1, None)
