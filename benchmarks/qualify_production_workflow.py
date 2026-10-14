"""Bounded production-workflow qualification, not a study of human productivity.

Run this script with the candidate wheel installed. The reference constructs a
SciPy LP directly from contract arrays and shares the HiGHS numerical kernel;
it is an independent formulation path, not an independent optimality checker.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy.optimize import linprog

from solverpilot.applications.production import (
    ProductionContract, replay_production_scenario, run_production_scenarios,
)
from solverpilot.runtime.manifest import _seal, _verify_integrity, environment, json_value

ATOL, RTOL = 1e-7, 1e-9


def _reject_constant(value):
    raise ValueError(f"nonfinite JSON number is not allowed: {value}")


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"), parse_constant=_reject_constant)


def _write_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_value(payload), stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")


def prepare_input(payload):
    if not isinstance(payload, dict) or payload.get("schema") != "solverpilot.production-pilot-input.v1":
        raise ValueError("unsupported production pilot input schema")
    allowed = {"schema", "label", "origin", "contract", "scenarios", "reference"}
    if set(payload) - allowed:
        raise ValueError("unknown pilot input fields")
    if not isinstance(payload.get("label"), str) or not payload["label"].strip():
        raise ValueError("pilot label must be nonempty")
    if not isinstance(payload.get("origin"), dict) or payload["origin"].get("kind") not in {
        "public_educational", "synthetic", "user_provided",
    }:
        raise ValueError("declare public_educational, synthetic or user_provided origin")
    if set(payload["origin"]) - {"kind", "url", "attribution"} or any(
        not isinstance(value, str) for value in payload["origin"].values()
    ):
        raise ValueError("origin allows only kind, url and attribution text fields")
    contract = ProductionContract.from_dict(payload["contract"])
    if not isinstance(payload.get("scenarios"), list):
        raise ValueError("scenarios must be a finite list")
    named, contracts, names = [], [("baseline", contract)], {"baseline"}
    for item in payload["scenarios"]:
        if not isinstance(item, dict) or set(item) != {"name", "updates"}:
            raise ValueError("each scenario requires name and updates only")
        name = item["name"]
        if not isinstance(name, str) or not name.strip() or name in names:
            raise ValueError("scenario names must be nonempty and unique; baseline is reserved")
        effective = contract.updated(item["updates"])
        names.add(name)
        named.append((name, item["updates"]))
        contracts.append((name, effective))
    reference = payload.get("reference", {})
    if not isinstance(reference, dict):
        raise ValueError("reference must be a mapping")
    expected = reference.get("baseline_objective")
    if expected is not None and (isinstance(expected, bool) or not isinstance(expected, (int, float))
                                 or not np.isfinite(expected)):
        raise ValueError("reference baseline objective must be finite")
    return contract, named, contracts, expected


def _close(left, right):
    return bool(left is not None and right is not None and np.isfinite(left) and np.isfinite(right)
                and abs(left - right) <= ATOL + RTOL * max(abs(left), abs(right)))


def independent_primal(contract, x, objective):
    """Direct raw-array check, independent of canonical/template validators."""
    if x is None or objective is None:
        return False
    x = np.asarray(x, dtype=float)
    if x.shape != contract.profit.shape or not np.isfinite(x).all():
        return False
    with np.errstate(over="ignore", invalid="ignore"):
        usage = contract.resources @ x
        profit = float(contract.profit @ x)
    finite_upper = np.isfinite(contract.maximum)
    return bool(np.isfinite(usage).all() and np.isfinite(profit)
                and np.all(x >= contract.minimum - ATOL - RTOL * np.abs(contract.minimum))
                and np.all(x[finite_upper] <= contract.maximum[finite_upper] + ATOL
                           + RTOL * np.abs(contract.maximum[finite_upper]))
                and np.all(usage <= contract.capacity + ATOL + RTOL * np.abs(contract.capacity))
                and _close(profit, objective))


def reference_solve(contract):
    """Never use build_model(), compile(), canonical conversion, or solve()."""
    start = perf_counter()
    c = -np.array(contract.profit, copy=True)
    A = np.array(contract.resources, copy=True)
    b = np.array(contract.capacity, copy=True)
    bounds = list(zip(contract.minimum.tolist(), contract.maximum.tolist()))
    built = perf_counter()
    result = linprog(c, A_ub=A, b_ub=b, bounds=bounds, method="highs-ds", options={"disp": False})
    solved = perf_counter()
    objective = None if result.x is None else float(contract.profit @ result.x)
    valid = independent_primal(contract, result.x, objective)
    return {"status": {0: "optimal", 1: "limit", 2: "infeasible", 3: "unbounded"}.get(result.status, "error"),
            "objective": objective, "primal_valid": valid,
            "build_s": built - start, "solve_s": solved - built,
            "validate_s": perf_counter() - solved}


def compare_case(case, contract, reference, expected=None, *, expected_name=None):
    summary = case.execution.summary if case.execution is not None else None
    status = None if summary is None else summary.status
    actual = None if summary is None else summary.objective
    own_primal = bool(summary is not None and independent_primal(contract, case.execution.result.x, actual))
    if reference["status"] == "optimal":
        status_agreement = bool(case.accepted and status == "valid_optimal")
        passed = bool(status_agreement and own_primal and reference["primal_valid"]
                      and _close(actual, reference["objective"]))
    elif reference["status"] in {"infeasible", "unbounded"}:
        status_agreement = status == reference["status"] and not case.accepted
        passed = status_agreement  # Numerical backend agreement only, never a new proof.
    else:
        status_agreement, passed = False, False
    expected_agreement = None if expected is None else _close(actual, expected) and _close(reference["objective"], expected)
    if expected_agreement is False:
        passed = False
    formulation_agreement = bool(case.formulation is not None and case.formulation.matches)
    contract_agreement = bool(case.contract is not None and case.contract.data_hash == contract.data_hash)
    name_agreement = expected_name is None or case.name == expected_name
    passed = bool(passed and formulation_agreement and contract_agreement and name_agreement)
    return {"name": case.name, "passed": bool(passed), "status_agreement": bool(status_agreement),
            "formulation_agreement": formulation_agreement, "contract_agreement": contract_agreement,
            "name_agreement": name_agreement,
            "solverpilot_status": status, "reference_status": reference["status"],
            "solverpilot_objective": actual, "reference_objective": reference["objective"],
            "independent_primal_check": own_primal, "expected_objective_agreement": expected_agreement,
            "optimality_evidence": None if summary is None else summary.optimality,
            "execution_id": None if summary is None else summary.run_id}


def replay_checks(study_path):
    start = perf_counter()
    recorded = read_json(study_path)
    _verify_integrity(recorded)
    if recorded.get("schema") != "solverpilot.production-study.v1":
        raise ValueError("unsupported study schema")
    if not recorded.get("input_complete") or not recorded["scenarios"]:
        raise ValueError("qualification requires a complete nonempty study")
    outcomes = []
    for row in recorded["scenarios"]:
        if "evidence" not in row or "contract" not in row:
            raise ValueError("every replay row requires full contract and execution evidence")
        result, formulation, semantic = replay_production_scenario(study_path, row["name"])
        old = row["evidence"]["summary"]
        identifier = row["evidence"]["execution_id"]
        linked = result.execution_id != identifier and result.trace.replay_of == identifier
        agrees = result.status.value == old["status"]
        if old["feasible"]:
            agrees = bool(agrees and semantic["feasible"] and semantic["objective_consistent"]
                          and _close(result.objective, old["objective"]))
        else:
            agrees = bool(agrees and not semantic["feasible"] and old["status"] in {"infeasible", "unbounded"})
        outcomes.append({"name": row["name"], "passed": bool(linked and formulation.matches and agrees),
                         "execution_id": result.execution_id, "replay_of": result.trace.replay_of,
                         "status": result.status.value, "objective": result.objective})
    return _seal({"schema": "solverpilot.production-pilot-replay.v1", "study_id": recorded["study_id"],
                  "passed": all(row["passed"] for row in outcomes), "elapsed_s": perf_counter() - start,
                  "environment": environment(), "source_and_versions_required": True,
                  "scenarios": outcomes})


def run_pilot(input_path, output, *, repeats=6, include_model=False):
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 2 <= repeats <= 100 or repeats % 2:
        raise ValueError("repeats must be an even integer between 2 and 100")
    output = Path(output)
    if output.exists():
        raise FileExistsError("refusing to overwrite a pilot output directory")
    start = perf_counter()
    payload = read_json(input_path)
    contract, scenarios, contracts, expected = prepare_input(payload)
    input_s = perf_counter() - start
    observations, agreement_rows = [], []
    study = None
    for iteration in range(repeats + 2):
        warmup = iteration < 2
        order = ("solverpilot", "reference") if iteration % 2 == 0 else ("reference", "solverpilot")
        clocks, references = {}, None
        for method in order:
            then = perf_counter()
            if method == "solverpilot":
                study = run_production_scenarios(contract, scenarios, backend="scipy-highs-ds", on_error="raise")
            else:
                references = [reference_solve(spec) for _, spec in contracts]
            clocks[method + "_wall_s"] = perf_counter() - then
        check_start = perf_counter()
        if len(study.scenarios) != len(contracts) or not study.input_complete:
            raise ValueError("pilot execution omitted input scenarios")
        comparisons = [compare_case(case, spec, ref, expected if i == 0 else None, expected_name=name)
                       for i, (case, (name, spec), ref) in enumerate(zip(study.scenarios, contracts, references))]
        agreement_rows.append({"iteration": iteration, "warmup": warmup, "cases": comparisons})
        clocks["comparison_s"] = perf_counter() - check_start
        observations.append({"iteration": iteration, "warmup": warmup, "order": list(order), **clocks,
            "solverpilot_phases": [{"name": case.name, **asdict(case.execution.result.trace.timings)}
                                   for case in study.scenarios], "reference_phases": references})
    output.mkdir(parents=True, exist_ok=False)
    render_start = perf_counter()
    report = study.render_markdown()
    render_s = perf_counter() - render_start
    export_start = perf_counter()
    study.save(output / "study.json", include_model=include_model)
    (output / "study.md").write_text(report, encoding="utf-8")
    if include_model:
        _write_new(output / "input.json", payload)
    export_s = perf_counter() - export_start
    replay = replay_checks(output / "study.json") if include_model else None
    if replay is not None:
        _write_new(output / "replay.json", replay)
    measured = [row for row in observations if not row["warmup"]]
    functional_passed = all(row["passed"] for trial in agreement_rows for row in trial["cases"])
    qualification = _seal({
        "schema": "solverpilot.production-pilot-qualification.v1", "label": payload["label"],
        "declared_origin": payload["origin"], "contract_hash": contract.data_hash,
        "environment": environment(), "source_sha256": study.scenarios[0].execution.result.trace.source_sha256,
        "reference_scope": "independent formulation; shared SciPy/HiGHS kernel; no new optimality certificate",
        "feasibility_atol": ATOL, "feasibility_rtol": RTOL,
        "timing_scope": "in-process warm-machine wall time; different work in the two paths; excludes human preparation and imports",
        "unmeasured_separate_stages": ["template/session compilation", "contract audit and diagnosis"],
        "unmeasured_stage_note": "included in aggregate workflow wall time; trace phases are not a complete breakdown",
        "input_read_validate_s": input_s, "report_render_s": render_s, "export_s": export_s,
        "median_solverpilot_workflow_s": median(row["solverpilot_wall_s"] for row in measured),
        "median_reference_s": median(row["reference_wall_s"] for row in measured),
        "measured_repetitions": repeats, "warmup_repetitions": 2,
        "observations": observations, "comparisons": agreement_rows,
        "reference_agreement_passed": functional_passed,
        "replay_passed": None if replay is None else replay["passed"],
        "technical_gate_passed": bool(functional_passed and replay is not None and replay["passed"]),
        "technical_gate_scope": "reference agreement and same-environment strict replay only",
        "clean_installation_qualified": False,
        "full_model_exported": include_model,
        "human_pilot": {"status": "awaiting_independent_participants", "feedback_records": 0,
                        "human_time_saving_established": False, "business_requirements_confirmed": False},
    })
    _write_new(output / "qualification.json", qualification)
    summary = (f"# Production pilot technical qualification\n\n{payload['label']}\n\n"
        f"- Reference agreement: {functional_passed}\n"
        f"- Replay passed: {qualification['replay_passed']}\n"
        f"- Technical gate passed: {qualification['technical_gate_passed']}\n"
        f"- Measured repetitions: {repeats}; warmups excluded: 2\n"
        f"- Median full workflow: {qualification['median_solverpilot_workflow_s']:.6f} s\n"
        f"- Median direct reference: {qualification['median_reference_s']:.6f} s\n\n"
        "These paths perform different work. This measures machine execution, not human time saved.\n"
        "Compilation and contract audits are included in aggregate workflow time, not measured separately.\n"
        "This gate alone does not qualify a clean wheel installation; that requires the separate installation check.\n"
        "The reference uses an independent formulation and the same SciPy/HiGHS kernel.\n"
        "Participant feedback and confirmation of real business requirements remain pending.\n")
    (output / "qualification.md").write_text(summary, encoding="utf-8")
    return qualification


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    run = commands.add_parser("run")
    run.add_argument("--input", type=Path, required=True)
    run.add_argument("--output", type=Path, required=True)
    run.add_argument("--repeats", type=int, default=6)
    run.add_argument("--include-model", action="store_true", help="Export full local inputs for strict replay; review before sharing")
    replay = commands.add_parser("replay")
    replay.add_argument("--study", type=Path, required=True)
    replay.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "run":
        result = run_pilot(args.input, args.output, repeats=args.repeats, include_model=args.include_model)
        passed = result["technical_gate_passed"]
    else:
        if args.output.exists():
            raise FileExistsError("refusing to overwrite replay evidence")
        result = replay_checks(args.study)
        _write_new(args.output, result)
        passed = result["passed"]
    print(json.dumps({"passed": passed, "output": str(args.output)}))
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
