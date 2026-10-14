"""Adversarial checks for the development reuse qualification, not a speed gate."""
from copy import deepcopy
from dataclasses import asdict
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from solverpilot.runtime.manifest import _seal
from solverpilot.trace import PhaseTimings


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("qualify_highs_reuse", ROOT / "benchmarks/qualify_highs_reuse.py")
challenge = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(challenge)


@pytest.fixture(scope="module")
def protocol():
    return challenge.build_protocol(seed=19, sizes=(1, 2), scales=(.5, 2.))


def _oracle_record(case, source, *, reuse):
    reference = challenge.analytic_reference(case)
    status = reference["status"]
    feasible = status == "optimal"
    retained = case["expected_reuse"] if reuse else False
    return {
        "outcome": "result", "backend_status": status,
        "status": "valid_optimal" if feasible else status,
        "x": reference["x"], "objective": reference["objective"], "validation_valid": feasible,
        "optimality_evidence": {"backend_reported_optimal": feasible, "primal_validated": feasible,
            "dual_verified": False, "gap_verified": False, "certificate_verified": False},
        "optimality_level": "solver_reported" if feasible else "not_established", "optimality_check": None,
        "execution_id": case["id"] + ("/warm" if reuse else "/cold"),
        "source_sha256": source, "problem_data_hash": challenge.build_problem(case).data_hash,
        "backend_version": "synthetic-test-record", "backend_configuration": challenge._options(case, reuse=reuse),
        "timings": asdict(PhaseTimings()), "call_wall_s": 0.,
        "native": {"model_status": status.title(), "primal_solution_status": "Feasible" if feasible else "None",
            "reuse_applied": retained, "reuse_reason": case["expected_reuse_reason"] if reuse else "reuse_disabled",
            "reuse_observations": {"native_model_retained": retained, "factorization_reuse": "not_observed"},
            "native_runtime_before_s": 0., "native_runtime_after_s": 0.,
            "requested_time_limit_s": case["data"]["time_limit_s"],
            "native_time_limit_s": case["data"]["time_limit_s"]},
    }


@pytest.fixture(scope="module")
def rows(protocol):
    source = protocol["source"]["package_source_sha256"]
    return [{"case_id": case["id"], "warm": _oracle_record(case, source, reuse=True),
             "cold": _oracle_record(case, source, reuse=False)} for case in protocol["cases"]]


def test_frozen_protocol_has_every_transition_and_both_presolve_modes(protocol):
    challenge.verify_protocol(protocol)
    assert protocol["classification"] == "development_stress_challenge"
    assert not protocol["untouched_heldout"]
    assert len(protocol["cases"]) == 2 * 2 * 2 * len(challenge.STEPS)
    sequences = {}
    for case in protocol["cases"]:
        sequences.setdefault(case["sequence_id"], []).append(case)
    for cases in sequences.values():
        assert tuple(case["step"] for case in cases) == challenge.STEPS
        assert cases[0]["data"]["presolve"] != cases[15]["data"]["presolve"]
    assert protocol == challenge.build_protocol(seed=19, sizes=(1, 2), scales=(.5, 2.))


@pytest.mark.parametrize("mutation", ["drop_case", "drop_presolve", "drop_limits", "loosen_tolerance", "heldout", "change_data", "overclaim_scope"])
def test_resealed_weakened_protocol_is_rejected(protocol, mutation):
    altered = deepcopy(protocol)
    if mutation == "drop_case":
        altered["cases"].pop()
    elif mutation == "drop_presolve":
        altered["settings"]["presolve"] = [False]
    elif mutation == "drop_limits":
        altered["cases"] = [case for case in altered["cases"] if not case["limited"]]
    elif mutation == "loosen_tolerance":
        altered["settings"]["atol"] = 1.
    elif mutation == "heldout":
        altered["untouched_heldout"] = True
    elif mutation == "overclaim_scope":
        altered["scope"] = "all LP models and native factorization reuse"
    else:
        altered["cases"][0]["data"]["cost"][0][0] += 1.
    with pytest.raises(ValueError):
        challenge.verify_protocol(_seal(altered))


def test_oracle_has_no_model_or_solver_dependency(protocol, monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("the analytic oracle used a model compiler or numerical optimizer")
    monkeypatch.setattr(challenge, "build_problem", forbidden)
    monkeypatch.setattr(challenge, "execute", forbidden)
    statuses = set()
    for case in protocol["cases"]:
        expected = challenge.analytic_reference(case)
        statuses.add(expected["status"])
        if expected["status"] == "optimal":
            assert challenge.independent_primal(case, expected["x"], expected["objective"])
    assert statuses == {"optimal", "infeasible", "unbounded"}


def test_oracle_matches_hand_calculated_pair_and_rejects_infeasible_point(protocol):
    case = deepcopy(protocol["cases"][0])
    case["data"].update(lower=[[0., 0.]], upper=[[4., 5.]], row_lower=[2.], row_upper=[6.],
                        cost=[[1., 3.]], row_scale=[1.], offset=7., sense="minimize")
    expected = challenge.analytic_reference(case)
    assert expected == {"status": "optimal", "objective": 9., "x": [2., 0.]}
    case["data"]["sense"] = "maximize"
    expected = challenge.analytic_reference(case)
    assert expected == {"status": "optimal", "objective": 23., "x": [1., 5.]}
    assert not challenge.independent_primal(case, [4., 5.], 26.)
    assert not challenge.independent_primal(case, [np.nan, 5.], 23.)


def test_degenerate_optimum_accepts_different_feasible_vectors(protocol):
    case = deepcopy(next(case for case in protocol["cases"] if case["step"] == "degenerate_optimum"))
    case["data"].update(lower=[[0., 0.]], upper=[[4., 5.]], row_lower=[0.], row_upper=[3.],
                        cost=[[-1., -1.]], row_scale=[1.], offset=0.)
    assert challenge.analytic_reference(case)["objective"] == -3.
    assert challenge.independent_primal(case, [3., 0.], -3.)
    assert challenge.independent_primal(case, [0., 3.], -3.)


def test_complete_records_pass_without_claiming_independent_proof_or_speed(protocol, rows):
    evaluation = challenge.evaluate_report(protocol, rows)
    assert evaluation["passed"] and evaluation["coverage_complete"]
    assert evaluation["case_count"] == len(protocol["cases"])
    assert evaluation["solve_count"] == 2 * len(protocol["cases"])
    assert not evaluation["speed_superiority_claim"]
    assert not evaluation["factorization_reuse_claim"]
    assert not evaluation["backend_only_termination_is_independent_proof"]
    for counts in evaluation["evidence_counts"].values():
        assert "independent_numerical_bound" not in counts


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "reorder", "drop_timeout"])
def test_gate_rejects_outcome_cherry_picking(protocol, rows, mutation):
    altered = deepcopy(rows)
    if mutation == "missing":
        altered.pop()
    elif mutation == "duplicate":
        altered[1] = deepcopy(altered[0])
    elif mutation == "reorder":
        altered[1], altered[2] = altered[2], altered[1]
    else:
        index = next(i for i, case in enumerate(protocol["cases"]) if case["limited"])
        altered.pop(index)
    evaluation = challenge.evaluate_report(protocol, altered)
    assert not evaluation["passed"] and not evaluation["coverage_complete"]


@pytest.mark.parametrize("mutation", [
    "wrong_objective", "wrong_primal", "stale_hash", "stale_options", "source", "identity",
    "invented_reuse", "inflated_level", "negative_timing", "double_count_timing", "timeout_rewritten",
])
def test_gate_detects_claim_and_transition_weakening(protocol, rows, mutation):
    altered = deepcopy(rows)
    record = altered[1]["warm"]
    if mutation == "wrong_objective":
        record["objective"] += 1.
    elif mutation == "wrong_primal":
        record["x"] = [0.] * len(record["x"])
    elif mutation == "stale_hash":
        record["problem_data_hash"] = altered[0]["warm"]["problem_data_hash"]
    elif mutation == "stale_options":
        record["backend_configuration"]["presolve"] = not record["backend_configuration"]["presolve"]
    elif mutation == "source":
        record["source_sha256"] = "0" * 64
    elif mutation == "identity":
        record["execution_id"] = altered[0]["warm"]["execution_id"]
    elif mutation == "invented_reuse":
        record["native"]["reuse_observations"]["factorization_reuse"] = "observed"
    elif mutation == "inflated_level":
        record["optimality_level"] = "independent_numerical_bound"
    elif mutation == "negative_timing":
        record["timings"]["certificate_s"] = -1.
    elif mutation == "double_count_timing":
        record["timings"].update(validate_s=1., primal_validate_s=1., certificate_s=1.)
    else:
        index = next(i for i, case in enumerate(protocol["cases"]) if case["limited"])
        altered[index]["warm"]["native"]["model_status"] = "Time limit reached"
    # A saved success flag cannot override failed observations.
    altered[1]["passed"] = True
    evaluation = challenge.evaluate_report(protocol, altered)
    assert not evaluation["passed"]
    assert evaluation["coverage_complete"]


def test_infeasible_status_cannot_be_promoted_to_proof(protocol, rows):
    altered = deepcopy(rows)
    index = next(i for i, case in enumerate(protocol["cases"]) if case["step"] == "infeasible")
    record = altered[index]["warm"]
    record["optimality_evidence"].update(dual_verified=True, gap_verified=True, certificate_verified=True)
    record["optimality_level"] = "independent_numerical_bound"
    assert not challenge.evaluate_report(protocol, altered)["passed"]


def test_limits_and_exceptions_are_retained_without_forcing_timeout_or_optimality(protocol, rows):
    altered = deepcopy(rows)
    index = next(i for i, case in enumerate(protocol["cases"]) if case["limited"])
    record = altered[index]["warm"]
    record.update(backend_status="limit_no_solution", status="error", x=None, objective=None,
                  validation_valid=False, optimality_level="not_established")
    record["optimality_evidence"].update(backend_reported_optimal=False, primal_validated=False)
    record["native"].update(model_status="Time limit reached", primal_solution_status="None")
    evaluation = challenge.evaluate_report(protocol, altered)
    assert evaluation["passed"]
    assert evaluation["case_count"] == len(rows)
    altered[index]["cold"] = {"outcome": "exception", "error_type": "RuntimeError", "error": "native failure"}
    evaluation = challenge.evaluate_report(protocol, altered)
    assert not evaluation["passed"] and evaluation["coverage_complete"]
    assert evaluation["case_count"] == len(rows)


def test_cli_freezes_inputs_before_solving_and_refuses_existing_output(protocol, rows, tmp_path, monkeypatch):
    output = tmp_path / "fresh"
    observed = []

    def run(cases):
        frozen = json.loads((output / "protocol.json").read_text())
        challenge.verify_protocol(frozen)
        assert frozen["cases"] == cases == protocol["cases"]
        observed.append(True)
        return rows

    monkeypatch.setattr(challenge, "run_cases", run)
    args = ["--output", str(output), "--seed", "19", "--sizes", "1", "2", "--scales", ".5", "2"]
    assert challenge.main(args) == 0
    assert observed == [True]
    saved = json.loads((output / "results.json").read_text())
    assert saved["evaluation"]["case_count"] == len(rows)
    with pytest.raises(FileExistsError):
        challenge.main(args)
    assert observed == [True]


def test_real_native_transition_sequence_in_fresh_process():
    if importlib.util.find_spec("highspy") is None:
        pytest.skip("optional highspy is unavailable")
    # One 20-case sequence is a bounded integration smoke test, not the main
    # multi-size/scale qualification. Isolate the process-global HiGHS scheduler.
    script = """
import importlib.util
from pathlib import Path
spec = importlib.util.spec_from_file_location('challenge', Path('benchmarks/qualify_highs_reuse.py'))
q = importlib.util.module_from_spec(spec); spec.loader.exec_module(q)
cases = q._sequence(19, 2, .5, False, 'native-smoke')
rows = q.run_cases(cases)
failures = []
for case, row in zip(cases, rows):
    for mode in ('warm', 'cold'):
        errors = q._record_errors(case, row[mode], reuse=mode == 'warm')
        if errors: failures.append((case['id'], mode, errors, row[mode]))
assert len(rows) == len(q.STEPS)
assert not failures, failures
"""
    env = dict(os.environ, PYTHONPATH=str(ROOT / "src"))
    completed = subprocess.run([sys.executable, "-c", script], cwd=ROOT, env=env,
                               text=True, capture_output=True, timeout=60)
    assert completed.returncode == 0, completed.stdout + completed.stderr
