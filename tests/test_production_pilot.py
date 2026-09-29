import importlib.util
from dataclasses import replace
import json
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("production_pilot", ROOT / "benchmarks/qualify_production_workflow.py")
pilot = importlib.util.module_from_spec(spec)
spec.loader.exec_module(pilot)


def input_payload():
    return {"schema": "solverpilot.production-pilot-input.v1", "label": "analytical fixture",
        "origin": {"kind": "synthetic"},
        "contract": {"profit": [40, 30], "resources": [[1, 1], [2, 1]], "capacity": [80, 100],
                     "minimum": [0, 0], "maximum": [40, "inf"]},
        "scenarios": [{"name": "more", "updates": {"capacity": [90, 100]}},
                      {"name": "tight", "updates": {"capacity": [70, 100]}},
                      {"name": "prices", "updates": {"profit": [30, 40]}},
                      {"name": "reset", "updates": {}}],
        "reference": {"baseline_objective": 2600}}


def write_input(tmp_path, payload=None):
    path = tmp_path / "input.json"
    path.write_text(json.dumps(input_payload() if payload is None else payload), encoding="utf-8")
    return path


def test_reference_does_not_compile_or_call_solverpilot():
    contract, _, _, _ = pilot.prepare_input(input_payload())
    with patch.object(pilot.ProductionContract, "build_model", side_effect=AssertionError("must not compile")), \
         patch.object(pilot, "run_production_scenarios", side_effect=AssertionError("must not use workflow")):
        result = pilot.reference_solve(contract)
    assert result["status"] == "optimal" and result["primal_valid"]
    assert result["objective"] == pytest.approx(2600)
    # A feasible point matches an independently derived exact upper bound.
    assert np.array([40, 30]) @ np.array([20, 60]) == 20 * 80 + 10 * 100 == 2600


@pytest.mark.parametrize("filename", ["nd-production.json", "synthetic-minimum-commitments.json"])
def test_distributed_inputs_are_executable_and_agree_with_stated_baseline(filename):
    payload = pilot.read_json(ROOT / "docs/pilot" / filename)
    contract, _, _, expected = pilot.prepare_input(payload)
    reference = pilot.reference_solve(contract)
    assert reference["primal_valid"] and reference["objective"] == pytest.approx(expected)
    assert payload["origin"]["kind"] in {"public_educational", "synthetic"}


def test_full_qualification_balances_order_and_replays(tmp_path):
    result = pilot.run_pilot(write_input(tmp_path), tmp_path / "output", repeats=2, include_model=True)
    assert result["technical_gate_passed"] and result["replay_passed"]
    assert result["human_pilot"]["feedback_records"] == 0
    assert not result["human_pilot"]["human_time_saving_established"]
    measured = [row for row in result["observations"] if not row["warmup"]]
    assert [row["order"] for row in measured] == [["solverpilot", "reference"], ["reference", "solverpilot"]]
    objectives = [row["solverpilot_objective"] for row in result["comparisons"][-1]["cases"]]
    assert objectives == pytest.approx([2600, 2800, 2400, 3200, 2600])
    replay = pilot.replay_checks(tmp_path / "output/study.json")
    assert replay["passed"] and all(row["execution_id"] != row["replay_of"] for row in replay["scenarios"])
    assert "human time saved" in (tmp_path / "output/qualification.md").read_text()


def test_summary_export_does_not_silently_export_inputs_or_qualify_replay(tmp_path):
    result = pilot.run_pilot(write_input(tmp_path), tmp_path / "summary", repeats=2)
    assert result["reference_agreement_passed"]
    assert result["replay_passed"] is None and not result["technical_gate_passed"]
    assert not (tmp_path / "summary/input.json").exists()
    study = pilot.read_json(tmp_path / "summary/study.json")
    assert all("contract" not in row and "model" not in row["evidence"]["run"] for row in study["scenarios"])
    with pytest.raises(ValueError, match="full contract"):
        pilot.replay_checks(tmp_path / "summary/study.json")


def test_infeasible_minimum_commitments_are_not_zero_profit(tmp_path):
    payload = input_payload()
    payload["contract"]["minimum"] = [10, 10]
    payload["scenarios"] = [{"name": "shortage", "updates": {"capacity": [5, 100]}}]
    result = pilot.run_pilot(write_input(tmp_path, payload), tmp_path / "shortage", repeats=2, include_model=True)
    assert result["technical_gate_passed"]
    shortfall = result["comparisons"][-1]["cases"][1]
    assert shortfall["solverpilot_status"] == shortfall["reference_status"] == "infeasible"
    assert shortfall["solverpilot_objective"] is None


def test_unbounded_case_keeps_termination_distinct_from_feasible_answer(tmp_path):
    payload = input_payload()
    payload["contract"] = {"profit": [1], "resources": [[0]], "capacity": [0]}
    payload["scenarios"], payload["reference"] = [], {}
    result = pilot.run_pilot(write_input(tmp_path, payload), tmp_path / "unbounded", repeats=2, include_model=True)
    assert result["technical_gate_passed"]
    assert result["comparisons"][-1]["cases"][0]["solverpilot_status"] == "unbounded"


def test_wrong_external_reference_fails_gate(tmp_path):
    payload = input_payload()
    payload["reference"]["baseline_objective"] = 2601
    result = pilot.run_pilot(write_input(tmp_path, payload), tmp_path / "wrong", repeats=2, include_model=True)
    assert not result["technical_gate_passed"]
    assert result["comparisons"][-1]["cases"][0]["expected_objective_agreement"] is False


def test_reference_status_error_cannot_pass_by_matching_objective(tmp_path, monkeypatch):
    original = pilot.reference_solve
    def failing(contract):
        return {**original(contract), "status": "error"}
    monkeypatch.setattr(pilot, "reference_solve", failing)
    result = pilot.run_pilot(write_input(tmp_path), tmp_path / "failed", repeats=2, include_model=True)
    assert not result["reference_agreement_passed"] and not result["technical_gate_passed"]


def test_nonunique_optima_do_not_require_same_decision_vector():
    contract = pilot.ProductionContract([1, 1], [[1, 1]], [1])
    assert pilot.independent_primal(contract, [1, 0], 1)
    assert pilot.independent_primal(contract, [0, 1], 1)
    study = pilot.run_production_scenarios(contract, backend="scipy-highs-ds")
    assert pilot.compare_case(study.scenarios[0], contract,
                              {"status": "optimal", "objective": 1, "primal_valid": True})["passed"]


def test_matching_infeasible_status_does_not_bypass_formulation_audit():
    contract = pilot.ProductionContract([1], [[1]], [0], minimum=[1])
    case = pilot.run_production_scenarios(contract, backend="scipy-highs-ds").scenarios[0]
    case = replace(case, formulation=replace(case.formulation, matches=False, issues=("wrong row",)))
    result = pilot.compare_case(case, contract, {"status": "infeasible", "objective": None})
    assert result["status_agreement"] and not result["passed"]


def test_coincident_objective_does_not_hide_swapped_scenario_contract_or_name():
    contract = pilot.ProductionContract([0], [[1]], [1])
    case = pilot.run_production_scenarios(contract, backend="scipy-highs-ds").scenarios[0]
    reference = {"status": "optimal", "objective": 0, "primal_valid": True}
    assert not pilot.compare_case(case, contract.updated({"capacity": [2]}), reference)["passed"]
    assert not pilot.compare_case(case, contract, reference, expected_name="another-case")["passed"]


def test_default_metadata_rejects_nested_private_data():
    payload = input_payload()
    payload["origin"]["private"] = {"contract": "SENTINEL_PRIVATE"}
    with pytest.raises(ValueError, match="origin allows"):
        pilot.prepare_input(payload)
    payload["origin"] = {"kind": "synthetic", "attribution": {"value": "SENTINEL_PRIVATE"}}
    with pytest.raises(ValueError, match="origin allows"):
        pilot.prepare_input(payload)


@pytest.mark.parametrize("x,objective", [([np.nan, 0], 0), ([np.inf, 0], 0), ([1], 1),
                                         ([-1, 1], 0), ([100, 100], 7000), ([20, 60], -2600)])
def test_independent_primal_rejects_invalid_candidates(x, objective):
    contract, _, _, _ = pilot.prepare_input(input_payload())
    assert not pilot.independent_primal(contract, x, objective)


@pytest.mark.parametrize("repeats", [True, 0, 1, 3, 102, 2.0])
def test_repetition_protocol_rejects_unbalanced_or_unbounded_counts(tmp_path, repeats):
    with pytest.raises(ValueError, match="even integer"):
        pilot.run_pilot(write_input(tmp_path), tmp_path / "bad", repeats=repeats)
    assert not (tmp_path / "bad").exists()


def test_input_validation_rejects_duplicates_and_nonfinite_json(tmp_path):
    payload = input_payload()
    payload["scenarios"].append({"name": "baseline", "updates": {}})
    with pytest.raises(ValueError, match="unique"):
        pilot.prepare_input(payload)
    path = write_input(tmp_path)
    path.write_text('{"value":NaN}')
    with pytest.raises(ValueError, match="nonfinite"):
        pilot.read_json(path)


def test_evidence_directory_cannot_be_overwritten(tmp_path):
    output = tmp_path / "old"
    output.mkdir()
    sentinel = output / "original.txt"
    sentinel.write_text("keep")
    with pytest.raises(FileExistsError):
        pilot.run_pilot(write_input(tmp_path), output, repeats=2)
    assert sentinel.read_text() == "keep"


def test_tampered_saved_study_is_rejected(tmp_path):
    output = tmp_path / "saved"
    pilot.run_pilot(write_input(tmp_path), output, repeats=2, include_model=True)
    path = output / "study.json"
    data = pilot.read_json(path)
    data["scenarios"][0]["name"] = "tampered"
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="hash"):
        pilot.replay_checks(path)
