import json
from dataclasses import replace

import pytest

from solverpilot import LinearProblem, solve
from solverpilot.history import HistoryStore
from solverpilot.reporting import explain_result
from solverpilot.runtime.unified import summarize
from solverpilot.runtime.manifest import run_manifest, save_run, replay_run, _seal
from solverpilot.runtime.evidence import (
    evidence_bundle, save_evidence_bundle, load_evidence_bundle,
    replay_evidence_bundle, verify_evidence_bundle,
)


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
        variable_upper=[2.], constraint_lower=[1.], constraint_upper=[float('inf')])


def test_one_execution_identity_links_all_core_artifacts(tmp_path):
    p = problem()
    result = solve(p, backend="scipy-highs-ds")
    identifier = result.execution_id
    assert identifier and summarize(result).run_id == summarize(result).run_id == identifier
    assert replace(result).execution_id == identifier
    assert replace(result.trace, termination="checked").execution_id == identifier
    report = explain_result(result)
    assert report.runtime["execution_id"] == identifier
    assert run_manifest(p, result)["execution_id"] == identifier
    with HistoryStore(tmp_path / "history.sqlite") as store:
        store.record_problem_object(p)
        first = store.record_solve_result(result)
        assert store.record_solve_result(result) == first
        alias = store.record_solve_result(result, run_id="external-alias")
        assert first.run_id == identifier and alias.run_id == "external-alias"
        assert alias.metadata["execution_id"] == identifier
    second = solve(p, backend="scipy-highs-ds")
    second = replace(second, trace=replace(second.trace, created_at_utc=result.trace.created_at_utc))
    assert second.execution_id != identifier
    assert second.trace.problem_data_hash == result.trace.problem_data_hash


def test_default_export_allowlist_and_explicit_full_optins(tmp_path):
    p = problem()
    original = solve(p)
    result = replace(original, raw_statistics={**original.raw_statistics,
        "private_debug_log": "SENTINEL_SECRET", "future_key": {"solution": [987654.]}},
        validation=replace(original.validation, warnings=("SENTINEL_SECRET",)),
        plan=replace(original.plan, rationale=("SENTINEL_SECRET",)),
        trace=replace(original.trace, warnings=("SENTINEL_SECRET",), parameters={
            **original.trace.parameters, "callback_path": "SENTINEL_SECRET"}))
    bundle = evidence_bundle(p, result)
    text = json.dumps(bundle)
    assert "SENTINEL_SECRET" not in text and "987654" not in text
    assert "model" not in bundle["run"] and bundle["run"]["statistics"] == {}
    assert bundle["run"]["parameters"] == {}
    assert bundle["summary"]["optimality"] == summarize(original).optimality
    assert "x" not in bundle["summary"]
    raw = run_manifest(p, result, include_raw_statistics=True)
    assert raw["statistics"]["private_debug_log"] == "SENTINEL_SECRET"
    assert "model" not in raw
    full = run_manifest(p, result, include_model=True)
    assert full["parameters"]["callback_path"] == "SENTINEL_SECRET"
    assert full["statistics"] == {}


def test_evidence_roundtrip_new_replay_identity_and_hash_checks(tmp_path):
    p = problem()
    result = solve(p, backend="scipy-highs-ds")
    path = tmp_path / "evidence.json"
    save_evidence_bundle(path, p, result, include_model=True)
    payload = load_evidence_bundle(path)
    replay = replay_evidence_bundle(path)
    assert replay.objective == pytest.approx(result.objective)
    assert replay.execution_id != result.execution_id
    assert replay.trace.replay_of == result.execution_id
    payload["summary"]["objective"] = 999.
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        load_evidence_bundle(path)
    payload = _seal(payload)
    with pytest.raises(ValueError, match="summaries"):
        verify_evidence_bundle(payload)


def test_inconsistent_bundle_identity_rejected_even_if_resealed():
    p = problem()
    bundle = evidence_bundle(p, solve(p, backend="scipy-highs-ds"))
    bundle["explanation"]["runtime"]["execution_id"] = "another-execution"
    with pytest.raises(ValueError, match="identity"):
        verify_evidence_bundle(_seal(bundle))


def test_source_mismatch_is_distinct_from_package_version(tmp_path):
    p = problem()
    path = tmp_path / "run.json"
    save_run(path, p, solve(p, backend="scipy-highs-ds"), include_model=True)
    payload = json.loads(path.read_text())
    assert len(payload["code_identity"]["source_sha256"]) == 64
    payload["code_identity"]["source_sha256"] = "0" * 64
    path.write_text(json.dumps(_seal(payload)))
    with pytest.raises(ValueError, match="source identity"):
        replay_run(path)
    assert replay_run(path, require_source=False).validation.valid


def test_legacy_v1_manifest_still_replays(tmp_path):
    p = problem()
    payload = run_manifest(p, solve(p, backend="scipy-highs-ds"), include_model=True)
    payload["schema"] = "solverpilot.run.v1"
    for key in ("execution_id", "code_identity", "integrity_sha256", "summary", "export_policy", "replay_of"):
        payload.pop(key)
    path = tmp_path / "legacy.json"
    path.write_text(json.dumps(payload))
    replay = replay_run(path)
    assert replay.validation.valid and replay.trace.replay_of is None
    with pytest.raises(ValueError, match="source identity"):
        replay_run(path, require_source=True)


def test_trace_less_exact_and_cp_results_keep_identity():
    from solverpilot.exact.runtime import ExactSolveResult
    from solverpilot.cp.reference import CPSolveResult
    for result in (ExactSolveResult("unavailable", False, "no checker"),
                   CPSolveResult("unknown", None, None, None, False, "fixture", {})):
        assert summarize(result).run_id == summarize(result).run_id == result.execution_id
        assert replace(result).execution_id == result.execution_id


def test_default_summary_cannot_be_replayed(tmp_path):
    p = problem()
    path = tmp_path / "summary.json"
    save_evidence_bundle(path, p, solve(p, backend="scipy-highs-ds"))
    with pytest.raises(ValueError, match="model data"):
        replay_evidence_bundle(path)


def test_source_unavailable_does_not_invent_identity_or_enable_replay(tmp_path):
    from solverpilot import _identity
    from unittest.mock import patch
    _identity.process_source_sha256.cache_clear()
    try:
        with patch.object(_identity, "source_tree_sha256", side_effect=OSError("unreadable")):
            p = problem()
            result = solve(p, backend="scipy-highs-ds")
            assert result.trace.source_sha256 is None
            path = tmp_path / "run.json"
            save_run(path, p, result, include_model=True)
            with pytest.raises(ValueError, match="source identity"):
                replay_run(path)
    finally:
        _identity.process_source_sha256.cache_clear()


@pytest.mark.parametrize("key,value", [("objective", 999.), ("backend", "another-backend"), ("status", "error")])
def test_contradictory_explanation_rejected_even_when_resealed(key, value):
    p = problem()
    bundle = evidence_bundle(p, solve(p, backend="scipy-highs-ds"))
    bundle["explanation"][key] = value
    with pytest.raises(ValueError, match="explanation"):
        verify_evidence_bundle(_seal(bundle))


def test_batch_sequential_preserves_underlying_execution(monkeypatch):
    import solverpilot.runtime.auto as auto
    from solverpilot.runtime.batch import iter_solve_batch
    captured = []
    original = auto.solve
    def capture(*args, **kwargs):
        result = original(*args, **kwargs)
        captured.append(result.execution_id)
        return result
    monkeypatch.setattr(auto, "solve", capture)
    items = list(iter_solve_batch([problem(), problem()], backend="scipy-highs-ds", mode="sequential"))
    assert [item.execution_id for item in items] == captured
    assert len(set(captured)) == 2


@pytest.mark.parametrize("field,value", [("status", "error"), ("backend", "another-backend"),
                                        ("schema", "solverpilot.run.v1")])
def test_inconsistent_nested_run_rejected_after_resealing(field, value):
    p = problem()
    bundle = evidence_bundle(p, solve(p, backend="scipy-highs-ds"))
    bundle["run"][field] = value
    bundle["run"] = _seal(bundle["run"])
    with pytest.raises(ValueError, match="run"):
        verify_evidence_bundle(_seal(bundle))


def test_inconsistent_source_link_rejected_after_resealing():
    p = problem()
    bundle = evidence_bundle(p, solve(p, backend="scipy-highs-ds"))
    bundle["explanation"]["runtime"]["source_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="source identity"):
        verify_evidence_bundle(_seal(bundle))


def test_isolated_worker_transports_identity(monkeypatch):
    from solverpilot.runtime.batch import _worker
    import solverpilot.runtime as runtime
    p = problem()
    result = solve(p, backend="scipy-highs-ds")
    class Capture:
        def send(self, value):
            self.payload = value
        def close(self):
            pass
    connection = Capture()
    monkeypatch.setattr(runtime, "solve", lambda *args, **kwargs: result)
    _worker(connection, p, "scipy-highs-ds", None, None, None)
    assert connection.payload["execution_id"] == result.execution_id


def test_invalid_nonfinite_candidate_remains_exportable_without_a_claim():
    import numpy as np
    from solverpilot import PublicStatus
    from solverpilot.validate import validate_solution, CandidateSolution
    p = problem()
    result = solve(p, backend="scipy-highs-ds")
    invalid = replace(result, status=PublicStatus.INVALID_SOLUTION, x=np.array([np.nan]),
        objective=None, validation=validate_solution(p, CandidateSolution(np.array([np.nan]), None)),
        raw_statistics={})
    payload = evidence_bundle(p, invalid)
    assert payload["explanation"]["schema_version"] == "unavailable"
    assert not payload["summary"]["feasible"]
    assert payload["explanation"]["claims"] == []
    verify_evidence_bundle(payload)
    json.dumps(payload, allow_nan=False)
