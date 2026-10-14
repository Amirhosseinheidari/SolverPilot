import json
from copy import deepcopy
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


@pytest.fixture(scope="module")
def recorded_evidence_examples():
    """Existing v1 wire shapes, including results without an independent bound."""
    import numpy as np
    from solverpilot import PublicStatus
    from solverpilot.backends import ScipyHighsBackend
    from solverpilot.validate import CandidateSolution, validate_solution

    p = problem()
    optimal = solve(p, backend="scipy-highs-ds")
    assert optimal.optimality_evidence.independently_verified_optimal
    examples = {"numerical": evidence_bundle(p, optimal),
                "backend_only": evidence_bundle(p, solve(p, backend=ScipyHighsBackend()))}
    milp = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[0.5], constraint_upper=[float("inf")], domains=["integer"])
    examples["milp"] = evidence_bundle(milp, solve(milp, backend=ScipyHighsBackend()))
    infeasible = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[1.],
        constraint_lower=[2.], constraint_upper=[float("inf")])
    examples["infeasible"] = evidence_bundle(infeasible, solve(infeasible, backend="scipy-highs-ds"))
    examples["confirmed_infeasible"] = evidence_bundle(infeasible,
        solve(infeasible, backend="scipy-highs-ds", diagnose_infeasible=True))
    unbounded = LinearProblem.from_data(A=[[0.]], c=[-1.], variable_lower=[0.],
        variable_upper=[float("inf")], constraint_lower=[float("-inf")], constraint_upper=[0.])
    examples["unbounded"] = evidence_bundle(unbounded, solve(unbounded, backend="scipy-highs-ds"))
    for name, status, backend_status in (
        ("feasible", PublicStatus.VALID_FEASIBLE, "converged_candidate"),
        ("limit", PublicStatus.FEASIBLE_LIMIT, "limit_feasible"),
        ("unknown_candidate", PublicStatus.UNKNOWN, "unknown"),
    ):
        examples[name] = evidence_bundle(p, replace(optimal, status=status, backend_status=backend_status,
                                                   raw_statistics={}))
    examples["error"] = evidence_bundle(p, replace(optimal, status=PublicStatus.ERROR, x=None,
        objective=None, validation=None, backend_status="solver_error", raw_statistics={}))
    for name, x in (("invalid", np.array([0.])), ("unavailable", np.array([np.nan]))):
        validation = validate_solution(p, CandidateSolution(x, None))
        examples[name] = evidence_bundle(p, replace(optimal, status=PublicStatus.INVALID_SOLUTION, x=x,
            objective=validation.objective_recomputed, validation=validation, raw_statistics={}))
    # An unavailable explanation can also result from nonfinite timing metadata;
    # it does not erase a separately recorded, valid summary.
    examples["unavailable_with_candidate"] = evidence_bundle(p, replace(optimal,
        trace=replace(optimal.trace, timings=replace(optimal.trace.timings, total_s=float("inf")))))
    return examples


@pytest.mark.parametrize("kind", ["numerical", "backend_only", "milp", "infeasible", "confirmed_infeasible",
    "unbounded", "feasible", "limit", "unknown_candidate", "error", "invalid", "unavailable", "unavailable_with_candidate"])
def test_existing_v1_evidence_shapes_remain_loadable(recorded_evidence_examples, kind, tmp_path):
    payload = deepcopy(recorded_evidence_examples[kind])
    assert payload["schema"] == "solverpilot.evidence.v1"
    # Wording is not a machine claim: existing report prose remains valid.
    payload["explanation"]["summary"] = "Previously recorded explanation wording."
    payload = _seal(payload)
    path = tmp_path / "legacy-evidence-v1.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    assert load_evidence_bundle(path) == payload


def _reseal_evidence(payload):
    payload["run"] = _seal(payload["run"])
    return _seal(payload)


@pytest.mark.parametrize("key", ["independently_verified_optimal", "primal_validated", "dual_verified",
                                     "gap_verified", "backend_reported_optimal"])
def test_resealed_optimality_flags_cannot_contradict_other_evidence(recorded_evidence_examples, key):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    assert payload["explanation"]["optimality"][key] is True
    payload["explanation"]["optimality"][key] = False
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("kind", ["backend_only", "milp", "infeasible", "error"])
def test_resealed_summary_cannot_promote_unverified_evidence(recorded_evidence_examples, kind):
    payload = deepcopy(recorded_evidence_examples[kind])
    assert payload["summary"]["optimality"] != "independent_numerical_bound"
    payload["summary"]["optimality"] = "independent_numerical_bound"
    payload["run"]["summary"]["optimality"] = "independent_numerical_bound"
    with pytest.raises(ValueError, match="optimality"):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("field,value", [
    ("valid", False), ("valid", 1), ("objective_consistent", "true"),
    ("objective_consistent", False), ("objective_consistent", None), ("objective_difference", 1.),
    ("objective_recomputed", None), ("objective_reported", None),
    ("objective_recomputed", 12345.), ("max_bound_violation", "zero"),
    ("max_constraint_violation", -1.), ("max_integrality_violation", True),
])
def test_resealed_validation_contradictions_and_types_rejected(recorded_evidence_examples, field, value):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    payload["explanation"]["validation"][field] = value
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("value", [1, 0, "true", None, [], {}])
def test_evidence_flags_require_actual_booleans(recorded_evidence_examples, value):
    for section, key in (("optimality", "independently_verified_optimal"), ("runtime", "reuse_applied")):
        if section == "runtime" and value is None:
            continue  # Unreported reuse has always been represented by null.
        payload = deepcopy(recorded_evidence_examples["numerical"])
        payload["explanation"][section][key] = value
        with pytest.raises(ValueError):
            verify_evidence_bundle(_reseal_evidence(payload))
    payload = deepcopy(recorded_evidence_examples["numerical"])
    payload["summary"]["feasible"] = payload["run"]["summary"]["feasible"] = value
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


def test_unknown_optimality_flags_are_not_a_supported_schema_extension(recorded_evidence_examples):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    payload["explanation"]["optimality"]["independent_exact_proof"] = True
    with pytest.raises(ValueError, match="optimality flags"):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("field,value", [("feasible", 1), ("objective", True)])
def test_nested_summary_boolean_number_equality_cannot_hide_malformed_fields(recorded_evidence_examples, field, value):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    assert payload["summary"][field] == value  # Equal in Python, but not the wire type.
    payload["run"]["summary"][field] = value
    with pytest.raises(ValueError, match="summary"):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("requested,elapsed,within", [
    (1., 2., True), (2., 1., False), (1., 1., False), (0., 0., False),
    (-1., 0., None), (1., -1., None), (None, -1., None),
])
def test_resealed_budget_values_cannot_contradict_recorded_times(recorded_evidence_examples, requested, elapsed, within):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    for summary in (payload["summary"], payload["run"]["summary"]):
        summary.update(requested_time_s=requested, elapsed_s=elapsed, within_budget=within)
    with pytest.raises(ValueError, match="summary"):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("requested,elapsed,within", [
    (1., 1., True), (1., 2., False), (2., 1., True), (0., 0., True),
    (None, 2., None), (1., None, None), (None, None, None),
    (None, 2., True), (1., None, False), (1., 2., None),
])
def test_recorded_budget_boundary_and_nullable_legacy_fields_remain_loadable(recorded_evidence_examples, requested, elapsed, within):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    for summary in (payload["summary"], payload["run"]["summary"]):
        summary.update(requested_time_s=requested, elapsed_s=elapsed, within_budget=within)
    payload = _reseal_evidence(payload)
    assert verify_evidence_bundle(payload) is payload


def test_actual_budgeted_solve_evidence_roundtrip(tmp_path):
    from solverpilot import SolveBudget
    p = problem()
    result = solve(p, backend="scipy-highs-ds", budget=SolveBudget(wall_time_s=5.))
    assert result.validation.valid
    path = tmp_path / "budgeted-evidence.json"
    save_evidence_bundle(path, p, result)
    summary = load_evidence_bundle(path)["summary"]
    assert summary["requested_time_s"] == 5.
    assert summary["elapsed_s"] >= 0
    assert summary["within_budget"] is (summary["elapsed_s"] <= summary["requested_time_s"])


@pytest.mark.parametrize("field,value", [("confirmed_infeasible", 1), ("iis_valid", "false"),
                                        ("iis_available", 0), ("conflict_irreducible", "true")])
def test_diagnostic_evidence_flags_are_typed(recorded_evidence_examples, field, value):
    payload = deepcopy(recorded_evidence_examples["confirmed_infeasible"])
    assert payload["explanation"]["diagnostics"] is not None
    payload["explanation"]["diagnostics"][field] = value
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("mutation", ["downgrade", "wrong_kind", "duplicate", "missing", "extra", "unknown_ref"])
def test_resealed_claims_must_agree_with_structured_evidence(recorded_evidence_examples, mutation):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    claims = payload["explanation"]["claims"]
    optimality = next(claim for claim in claims if claim["claim_id"] == "solution.optimality")
    if mutation == "downgrade":
        optimality["disposition"] = "blocked"
    elif mutation == "wrong_kind":
        optimality["kind"] = "feasibility"
    elif mutation == "duplicate":
        claims.append(deepcopy(optimality))
    elif mutation == "missing":
        claims.remove(optimality)
    elif mutation == "extra":
        claims.append({**optimality, "claim_id": "solution.unrecorded_proof"})
    else:
        optimality["evidence_refs"] = ["result.untrusted_proof"]
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("kind,claim_id", [("infeasible", "solution.infeasibility"),
                                         ("unbounded", "solution.unboundedness"),
                                         ("error", "solution.feasibility")])
def test_qualified_or_blocked_claims_cannot_be_promoted_without_evidence(recorded_evidence_examples, kind, claim_id):
    payload = deepcopy(recorded_evidence_examples[kind])
    claim = next(claim for claim in payload["explanation"]["claims"] if claim["claim_id"] == claim_id)
    claim["disposition"] = "supported"
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("field,value", [("schema_version", "future"), ("schema_version", None),
    ("schema_version", {}), ("claims", {}), ("claims", [None]), ("validation", []),
    ("optimality", []), ("runtime", None), ("diagnostics", []), ("planner", [])])
def test_malformed_explanation_shapes_raise_value_error(recorded_evidence_examples, field, value):
    payload = deepcopy(recorded_evidence_examples["numerical"])
    payload["explanation"][field] = value
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


@pytest.mark.parametrize("field", ["optimality", "claims", "validation", "planner", "diagnostics", "runtime"])
def test_unavailable_explanation_cannot_smuggle_available_claims(recorded_evidence_examples, field):
    payload = deepcopy(recorded_evidence_examples["unavailable"])
    if field == "runtime":
        payload["explanation"]["runtime"]["reuse_applied"] = True
    elif field == "claims":
        payload["explanation"][field] = deepcopy(recorded_evidence_examples["error"]["explanation"][field])
    elif field == "optimality":
        payload["explanation"][field] = {"independently_verified_optimal": False}
    else:
        payload["explanation"][field] = {}
    with pytest.raises(ValueError):
        verify_evidence_bundle(_reseal_evidence(payload))


def test_recorded_verification_does_not_rerun_solver_or_authenticate(recorded_evidence_examples, monkeypatch):
    def unexpected_solve(*args, **kwargs):
        pytest.fail("loading recorded evidence must not claim to recertify it")
    monkeypatch.setattr("solverpilot.runtime.auto.solve", unexpected_solve)
    assert verify_evidence_bundle(recorded_evidence_examples["numerical"]) is recorded_evidence_examples["numerical"]
