"""Replayable development stress challenge for opt-in native HiGHS LP reuse.

This is a synthetic, development-visible protocol, not untouched heldout evidence
or a performance-superiority study. The reference enumerates the vertices of
independent two-variable boxes intersected by row strips. It never consumes a
compiled problem or calls an optimizer. Presolve on/off and every transition are
mandatory; all outcomes, including exceptions and native limits, are retained.
Content hashes identify the recorded source and inputs; they do not authenticate
an author or certify arbitrary edited result files.
"""
from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, execute
from solverpilot._identity import process_source_sha256
from solverpilot.backends import HighspyNativeBackend
from solverpilot.runtime.manifest import _seal, _verify_integrity, environment, json_value
from solverpilot.runtime.unified import summarize


PROTOCOL_SCHEMA = "solverpilot.highs-reuse-development-protocol.v1"
RESULT_SCHEMA = "solverpilot.highs-reuse-development-results.v1"
ATOL, RTOL = 1e-7, 1e-9
REFERENCE = "closed-form vertices and recession directions of separable two-variable row strips"
SCOPE = "continuous LP; synthetic independent two-variable blocks; no factorization or speed-superiority claim"
STEPS = (
    "baseline", "cost_switch", "row_bounds", "column_bounds", "upper_row_active",
    "offset", "degenerate_optimum", "maximize", "maximize_updates", "infeasible",
    "feasible_again", "unbounded", "bounded_again", "matrix_values_rebuild",
    "structure_rebuild", "configuration_rebuild", "after_configuration_update",
    "tiny_limit", "tiny_limit_retained_update", "normal_limit_restored",
)


def _source():
    return {"package_source_sha256": process_source_sha256(),
            "challenge_script_sha256": sha256(Path(__file__).read_bytes()).hexdigest()}


def _settings(seed, sizes, scales):
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError("seed must be an integer in [0, 2**32)")
    sizes, scales = list(sizes), list(scales)
    if (not 2 <= len(sizes) <= 3 or len(set(sizes)) != len(sizes)
            or any(type(n) is not int or not 1 <= n <= 64 for n in sizes)):
        raise ValueError("use two or three distinct block sizes in [1, 64]")
    if (not 2 <= len(scales) <= 3 or len(set(scales)) != len(scales)
            or any(type(s) not in (int, float) or not np.isfinite(s) or not 1e-3 <= s <= 1e3 for s in scales)):
        raise ValueError("use two or three distinct finite scales in [1e-3, 1e3]")
    return {"seed": seed, "sizes": sizes, "scales": [float(s) for s in scales],
            "presolve": [False, True], "solver": "simplex", "threads": 1,
            "tiny_native_limit_s": 1e-12, "atol": ATOL, "rtol": RTOL}


def _sequence(seed, blocks, scale, presolve, sequence_id):
    rng = np.random.default_rng(np.random.SeedSequence([seed, blocks]))
    units = rng.integers(1, 9, size=blocks + 1) / 2.
    costs = rng.integers(1, 5, size=blocks + 1) * scale
    state = {
        "lower": np.zeros((blocks, 2)), "upper": units[:blocks, None] * np.array([4., 5.]),
        "row_lower": 2. * units[:blocks], "row_upper": 6. * units[:blocks],
        "cost": costs[:blocks, None] * np.array([1., 3.]),
        "row_scale": np.full(blocks, scale), "sense": "minimize", "offset": 0.,
        "presolve": presolve, "time_limit_s": None,
    }
    cases = []

    def add(step, *, reuse=True, reason="fixed_matrix_vector_update"):
        cases.append(json_value({"id": f"{sequence_id}/{step}", "sequence_id": sequence_id,
            "step": step, "data": state, "expected_reuse": reuse,
            "expected_reuse_reason": reason, "limited": state["time_limit_s"] is not None}))

    add("baseline", reuse=False, reason="first_solve")
    state["cost"] = costs[:blocks, None] * np.array([3., 1.]); add("cost_switch")
    state["row_lower"] = 5. * units[:blocks]
    state["row_upper"] = 7. * units[:blocks]; add("row_bounds")
    state["upper"] = units[:blocks, None] * np.array([4., 1.]); add("column_bounds")
    state["upper"] = units[:blocks, None] * np.array([4., 5.])
    state["cost"] = costs[:blocks, None] * np.array([-3., -1.])
    state["row_lower"] = np.zeros(blocks)
    state["row_upper"] = 3. * units[:blocks]; add("upper_row_active")
    state["offset"] = float((seed % 17 + 1) * scale); add("offset")
    state["cost"] = costs[:blocks, None] * np.array([-1., -1.]); add("degenerate_optimum")
    state["sense"] = "maximize"
    state["cost"] = costs[:blocks, None] * np.array([3., 1.])
    state["row_lower"] = 2. * units[:blocks]
    state["row_upper"] = 6. * units[:blocks]
    add("maximize", reuse=False, reason="objective_sense_changed")
    state["cost"] = costs[:blocks, None] * np.array([1., 3.])
    state["upper"] = units[:blocks, None] * np.array([4., 2.])
    state["row_upper"] = 5. * units[:blocks]
    state["offset"] *= -2.; add("maximize_updates")
    state["row_lower"] = 7. * units[:blocks]
    state["row_upper"] = 9. * units[:blocks]; add("infeasible")
    state["row_lower"] = 2. * units[:blocks]
    state["row_upper"] = 5. * units[:blocks]; add("feasible_again")
    state["upper"] = np.full((blocks, 2), np.inf)
    state["row_upper"] = np.full(blocks, np.inf); add("unbounded")
    state["upper"] = units[:blocks, None] * np.array([4., 2.])
    state["row_upper"] = 5. * units[:blocks]; add("bounded_again")
    state["row_scale"] = np.full(blocks, 2. * scale)
    add("matrix_values_rebuild", reuse=False, reason="matrix_values_changed")
    state.update(lower=np.zeros((blocks + 1, 2)), upper=units[:, None] * np.array([4., 2.]),
                 row_lower=2. * units, row_upper=5. * units,
                 cost=costs[:, None] * np.array([1., 3.]), row_scale=np.full(blocks + 1, 2. * scale))
    add("structure_rebuild", reuse=False, reason="structure_or_domains_changed")
    state["presolve"] = not presolve
    add("configuration_rebuild", reuse=False, reason="configuration_changed")
    state["cost"] = costs[:, None] * np.array([3., 1.]); add("after_configuration_update")
    state["time_limit_s"] = 1e-12
    state["presolve"] = not state["presolve"]
    add("tiny_limit", reuse=False, reason="configuration_changed")
    state["cost"] = costs[:, None] * np.array([1., 3.]); add("tiny_limit_retained_update")
    state["time_limit_s"] = None
    add("normal_limit_restored", reason="unchanged_problem")
    assert tuple(case["step"] for case in cases) == STEPS
    return cases


def _cases(settings):
    cases = []
    for blocks in settings["sizes"]:
        for scale in settings["scales"]:
            for presolve in settings["presolve"]:
                sequence_id = f"blocks-{blocks}_scale-{scale!r}_presolve-{int(presolve)}"
                cases.extend(_sequence(settings["seed"], blocks, scale, presolve, sequence_id))
    return cases


def build_protocol(*, seed=73129, sizes=(4, 16), scales=(.01, 1., 100.)):
    settings = _settings(seed, sizes, scales)
    return _seal({"schema": PROTOCOL_SCHEMA, "classification": "development_stress_challenge",
        "untouched_heldout": False, "settings": settings, "source": _source(),
        "reference": REFERENCE, "scope": SCOPE,
        "cases": _cases(settings)})


def verify_protocol(protocol, *, require_source=True):
    _verify_integrity(protocol)
    if protocol.get("schema") != PROTOCOL_SCHEMA:
        raise ValueError("unsupported reuse protocol schema")
    if (set(protocol) != {"schema", "classification", "untouched_heldout", "settings", "source",
                          "reference", "scope", "cases", "integrity_sha256"}
            or protocol.get("reference") != REFERENCE or protocol.get("scope") != SCOPE):
        raise ValueError("protocol scope or reference claims changed")
    settings = protocol["settings"]
    expected_settings = _settings(settings["seed"], settings["sizes"], settings["scales"])
    if settings != expected_settings or protocol.get("cases") != _cases(expected_settings):
        raise ValueError("protocol coverage or generated case data changed")
    if protocol.get("classification") != "development_stress_challenge" or protocol.get("untouched_heldout") is not False:
        raise ValueError("development cases cannot be labeled untouched heldout")
    if require_source and protocol.get("source") != _source():
        raise ValueError("protocol source differs from this package or challenge script")
    if (set(protocol["source"]) != {"package_source_sha256", "challenge_script_sha256"}
            or any(not isinstance(value, str) or len(value) != 64
                   or any(char not in "0123456789abcdef" for char in value)
                   for value in protocol["source"].values())):
        raise ValueError("protocol requires complete source digests")


def _arrays(case):
    data = case["data"]
    return tuple(np.asarray(data[key], dtype=float) for key in
                 ("lower", "upper", "row_lower", "row_upper", "cost", "row_scale"))


def analytic_reference(case):
    """An algebraic oracle on raw case data, independent of model construction.

    All lower variable bounds are finite. Feasibility follows from overlapping
    attainable row-sum intervals. An improving coordinate ray exists exactly
    when its upper bound and the row upper bound are both infinite. Otherwise
    a linear optimum lies at one of the explicit box/row-strip vertices.
    """
    lower, upper, row_lower, row_upper, costs, _ = _arrays(case)
    sign = 1. if case["data"]["sense"] == "minimize" else -1.
    if np.any(lower.sum(axis=1) > row_upper) or np.any(upper.sum(axis=1) < row_lower):
        return {"status": "infeasible", "objective": None, "x": None}
    if np.any((sign * costs < 0) & np.isposinf(upper) & np.isposinf(row_upper[:, None])):
        return {"status": "unbounded", "objective": None, "x": None}
    solution = []
    for lo, hi, rlo, rhi, cost in zip(lower, upper, row_lower, row_upper, costs):
        vertices = []
        for j in (0, 1):
            for endpoint in (lo[j], hi[j]):
                if not np.isfinite(endpoint):
                    continue
                for other in (lo[1-j], hi[1-j], rlo-endpoint, rhi-endpoint):
                    if not np.isfinite(other):
                        continue
                    point = np.array([endpoint, other] if j == 0 else [other, endpoint])
                    if np.all(point >= lo) and np.all(point <= hi) and rlo <= point.sum() <= rhi:
                        vertices.append(point)
        if not vertices:
            raise ValueError("feasible block has no enumerated vertex")
        solution.extend(min(vertices, key=lambda x: sign * float(cost @ x)))
    x = np.asarray(solution)
    return {"status": "optimal", "objective": float(costs.reshape(-1) @ x + case["data"]["offset"]),
            "x": x.tolist()}


def build_problem(case):
    lower, upper, row_lower, row_upper, costs, row_scale = _arrays(case)
    n = len(lower)
    matrix = sparse.csr_matrix((np.repeat(row_scale, 2),
        (np.repeat(np.arange(n), 2), np.arange(2*n))), shape=(n, 2*n))
    return LinearProblem.from_data(A=matrix, c=costs.reshape(-1),
        variable_lower=lower.reshape(-1), variable_upper=upper.reshape(-1),
        constraint_lower=row_scale * row_lower, constraint_upper=row_scale * row_upper,
        objective_sense=case["data"]["sense"], objective_offset=case["data"]["offset"])


def _close(a, b):
    return bool(type(a) in (int, float) and type(b) in (int, float)
                and np.isfinite([a, b]).all() and abs(a-b) <= ATOL + RTOL * max(abs(a), abs(b)))


def independent_primal(case, x, objective):
    if x is None:
        return False
    lower, upper, row_lower, row_upper, costs, _ = _arrays(case)
    try:
        values = np.asarray(x, dtype=float)
    except (TypeError, ValueError):
        return False
    if values.shape != (lower.size,) or not np.isfinite(values).all():
        return False
    values = values.reshape(lower.shape)
    sums = values.sum(axis=1)

    def above(actual, bound):
        mask = np.isfinite(bound)
        return bool(np.all(actual[mask] >= bound[mask] - ATOL - RTOL * np.maximum(1., np.abs(bound[mask]))))

    return (above(values, lower) and above(-values, -upper)
            and above(sums, row_lower) and above(-sums, -row_upper)
            and _close(float((costs * values).sum() + case["data"]["offset"]), objective))


def _options(case, *, reuse):
    return {"reuse": reuse, "solver": "simplex", "threads": 1,
            "presolve": case["data"]["presolve"], "time_limit_s": case["data"]["time_limit_s"]}


def _capture(case, backend):
    started = perf_counter()
    try:
        result = execute(build_problem(case), backend)
        elapsed = perf_counter() - started
        raw = result.raw_statistics or {}
        return json_value({"outcome": "result", "backend_status": result.backend_status,
            "status": result.status.value, "x": result.x, "objective": result.objective,
            "validation_valid": bool(result.validation and result.validation.valid),
            "optimality_evidence": asdict(result.optimality_evidence),
            "optimality_level": summarize(result).optimality,
            "optimality_check": raw.get("optimality_check"),
            "execution_id": result.execution_id, "problem_data_hash": result.trace.problem_data_hash,
            "source_sha256": result.trace.source_sha256,
            "backend_version": result.trace.backend_version,
            "backend_configuration": result.trace.parameters["backend_configuration"],
            "timings": asdict(result.trace.timings), "call_wall_s": elapsed,
            "native": {key: raw.get(key) for key in (
                "model_status", "primal_solution_status", "simplex_iteration_count",
                "reuse_applied", "reuse_mode", "reuse_reason", "updated_fields", "reuse_observations",
                "requested_time_limit_s", "native_time_limit_s", "native_runtime_before_s", "native_runtime_after_s")}})
    except Exception as exc:  # Each failure is a retained, failing observation.
        return {"outcome": "exception", "error_type": type(exc).__name__, "error": str(exc),
                "call_wall_s": perf_counter()-started}


def run_cases(cases):
    """Run every pair, retaining state only within a protocol sequence."""
    rows, warm, sequence_id = [], None, None
    try:
        for case in cases:
            if case["sequence_id"] != sequence_id:
                if warm is not None:
                    warm.close()
                warm = HighspyNativeBackend(**_options(case, reuse=True))
                sequence_id = case["sequence_id"]
            for key, value in _options(case, reuse=True).items():
                setattr(warm, key, value)
            # Alternate execution order without changing the retained sequence.
            with HighspyNativeBackend(**_options(case, reuse=False)) as cold:
                if len(rows) % 2:
                    cold_result, warm_result = _capture(case, cold), _capture(case, warm)
                else:
                    warm_result, cold_result = _capture(case, warm), _capture(case, cold)
            rows.append({"case_id": case["id"], "warm": warm_result, "cold": cold_result})
    finally:
        if warm is not None:
            warm.close()
    return rows


def _record_errors(case, record, *, reuse):
    errors = []
    if record.get("outcome") != "result":
        return ["solve exception or missing result"]
    reference = analytic_reference(case)
    status = record.get("backend_status")
    expected = reference["status"]
    allowed = {"optimal", "limit_feasible", "limit_no_solution"} if case["limited"] else {expected}
    if status not in allowed:
        errors.append("unexpected backend status")
    x, objective = record.get("x"), record.get("objective")
    valid = independent_primal(case, x, objective)
    primal_claim = record.get("validation_valid") is True
    if x is not None and (not valid or not primal_claim):
        errors.append("candidate failed direct or canonical primal validation")
    if status in {"optimal", "limit_feasible"} and not valid:
        errors.append("feasible termination has no valid candidate")
    if status not in {"optimal", "limit_feasible"} and (x is not None or objective is not None or primal_claim):
        errors.append("non-solution termination retained a candidate or feasibility claim")
    if status == "optimal" and not _close(objective, reference["objective"]):
        errors.append("objective differs from analytic optimum")
    if valid and expected == "optimal":
        sign = 1. if case["data"]["sense"] == "minimize" else -1.
        if sign * (objective-reference["objective"]) < -ATOL-RTOL*abs(reference["objective"]):
            errors.append("candidate improves on the analytic optimum")
    public = {"optimal": "valid_optimal", "limit_feasible": "feasible_limit",
              "limit_no_solution": "error", "infeasible": "infeasible", "unbounded": "unbounded"}.get(status)
    if record.get("status") != public:
        errors.append("public status does not preserve native termination")

    trust = record.get("optimality_evidence", {})
    flags = ("backend_reported_optimal", "primal_validated", "dual_verified", "gap_verified", "certificate_verified")
    if set(trust) != set(flags) or any(type(trust.get(key)) is not bool for key in flags):
        errors.append("malformed evidence flags")
    if trust.get("backend_reported_optimal") is not (status == "optimal") or trust.get("primal_validated") is not primal_claim:
        errors.append("evidence disagrees with recorded termination or primal validation")
    check = record.get("optimality_check") or {}
    independent = bool(trust.get("primal_validated") and trust.get("dual_verified")
                       and (trust.get("gap_verified") or trust.get("certificate_verified")))
    if (trust.get("dual_verified") is not (check.get("dual_valid") is True)
            or trust.get("gap_verified") is not (check.get("verified") is True)
            or trust.get("certificate_verified") is not False):
        errors.append("evidence exceeds the recorded numerical check")
    level = "independent_numerical_bound" if independent else (
        "solver_reported" if primal_claim and status == "optimal" else "not_established")
    if record.get("optimality_level") != level:
        errors.append("inflated or inconsistent optimality level")
    if independent and (not valid or not _close(objective, reference["objective"])):
        errors.append("independent optimality claim failed analytic comparison")
    if expected in {"infeasible", "unbounded"} and (independent or level != "not_established"):
        errors.append("backend-only termination promoted to an independent proof")

    native = record.get("native", {})
    model_status = str(native.get("model_status", "")).casefold()
    primal_status = str(native.get("primal_solution_status", "")).casefold()
    observed_status = None
    if "optimal" in model_status:
        observed_status = "optimal"
    elif "infeasible" in model_status and "unbounded" in model_status:
        observed_status = "infeasible_or_unbounded"
    elif "infeasible" in model_status:
        observed_status = "infeasible"
    elif "unbounded" in model_status:
        observed_status = "unbounded"
    elif any(token in model_status for token in ("time limit", "iteration limit", "solution limit", "objective bound", "objective target", "interrupt")):
        observed_status = "limit_feasible" if "feasible" in primal_status and "infeasible" not in primal_status else "limit_no_solution"
    if status != observed_status:
        errors.append("backend termination differs from the native status strings")
    expected_reuse = case["expected_reuse"] if reuse else False
    expected_reason = case["expected_reuse_reason"] if reuse else "reuse_disabled"
    if native.get("reuse_applied") is not expected_reuse or native.get("reuse_reason") != expected_reason:
        errors.append("retention or rebuild does not match the transition")
    observations = native.get("reuse_observations") or {}
    if observations.get("native_model_retained") is not expected_reuse or observations.get("factorization_reuse") != "not_observed":
        errors.append("native reuse observations missing or overstated")
    if record.get("backend_configuration") != _options(case, reuse=reuse):
        errors.append("backend options differ from the frozen case")
    if record.get("problem_data_hash") != build_problem(case).data_hash:
        errors.append("solved problem differs from the frozen case")
    if not isinstance(record.get("backend_version"), str) or not record["backend_version"]:
        errors.append("native version not recorded")
    native_before = native.get("native_runtime_before_s")
    native_after = native.get("native_runtime_after_s")
    if native.get("requested_time_limit_s") != case["data"]["time_limit_s"]:
        errors.append("native requested allowance differs from the frozen case")
    if (type(native_before) not in (int, float) or type(native_after) not in (int, float)
            or not np.isfinite([native_before, native_after]).all() or not 0 <= native_before <= native_after):
        errors.append("invalid native cumulative runtime observations")
    elif case["limited"]:
        if native.get("native_time_limit_s") != native_before + case["data"]["time_limit_s"]:
            errors.append("native limit is not the recorded per-run allowance")
    elif native.get("native_time_limit_s") is not None:
        errors.append("unexpected native time limit")
    timings = record.get("timings", {})
    required = ("backend_build_s", "backend_update_s", "solve_s", "backend_total_s",
                "primal_validate_s", "certificate_s", "validate_s", "total_s")
    if any(type(timings.get(key)) not in (int, float) or not np.isfinite(timings[key]) or timings[key] < 0 for key in required):
        errors.append("missing or invalid phase timings")
    else:
        if not _close(timings["validate_s"], timings["primal_validate_s"] + timings["certificate_s"]):
            errors.append("validation stages are not additive")
        if sum(timings[key] for key in ("backend_build_s", "backend_update_s", "solve_s")) > timings["backend_total_s"] + 1e-7:
            errors.append("native stages exceed the backend call")
        if timings["backend_total_s"] + timings["validate_s"] > timings["total_s"] + 1e-7:
            errors.append("backend and validation stages exceed the execution")
        elapsed = record.get("call_wall_s")
        if type(elapsed) not in (int, float) or not np.isfinite(elapsed) or elapsed < timings["total_s"] - 1e-7:
            errors.append("call wall time does not enclose execution")
    return errors


def evaluate_report(protocol, rows):
    """Recompute the gate from retained observations, never saved pass flags."""
    verify_protocol(protocol)
    expected_ids = [case["id"] for case in protocol["cases"]]
    if not isinstance(rows, list) or [row.get("case_id") for row in rows] != expected_ids:
        return {"passed": False, "coverage_complete": False,
                "errors": ["missing, duplicate, unexpected or reordered case outcomes"], "cases": []}
    results, evidence_counts, execution_ids = [], {"warm": Counter(), "cold": Counter()}, set()
    for case, row in zip(protocol["cases"], rows):
        errors = []
        for mode in ("warm", "cold"):
            record = row.get(mode, {})
            errors.extend(f"{mode}: {error}" for error in _record_errors(case, record, reuse=mode == "warm"))
            evidence_counts[mode][record.get("optimality_level", "no_result")] += 1
            if record.get("outcome") == "result" and record.get("source_sha256") != protocol["source"]["package_source_sha256"]:
                errors.append(f"{mode}: execution source differs from protocol")
            if record.get("outcome") == "result":
                identity = record.get("execution_id")
                if not isinstance(identity, str) or not identity or identity in execution_ids:
                    errors.append(f"{mode}: missing or reused execution identity")
                else:
                    execution_ids.add(identity)
        warm, cold = row.get("warm", {}), row.get("cold", {})
        if not case["limited"] and warm.get("outcome") == cold.get("outcome") == "result":
            if warm.get("backend_status") != cold.get("backend_status"):
                errors.append("cold and retained terminations disagree")
            if warm.get("optimality_level") != cold.get("optimality_level"):
                errors.append("cold and retained evidence levels disagree")
        results.append({"case_id": case["id"], "passed": not errors, "errors": errors,
                        "analytic_status": analytic_reference(case)["status"]})
    return {"passed": all(row["passed"] for row in results), "coverage_complete": True,
            "case_count": len(rows), "solve_count": 2*len(rows),
            "evidence_counts": {mode: dict(counts) for mode, counts in evidence_counts.items()},
            "limited_case_count": sum(case["limited"] for case in protocol["cases"]),
            "speed_superiority_claim": False, "factorization_reuse_claim": False,
            "backend_only_termination_is_independent_proof": False, "cases": results}


def _write_new(path, payload):
    with Path(path).open("x", encoding="utf-8", newline="\n") as stream:
        json.dump(json_value(payload), stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True, help="new output directory; existing paths are rejected")
    parser.add_argument("--seed", type=int, default=73129)
    parser.add_argument("--sizes", type=int, nargs="+", default=[4, 16], help="two or three block sizes, each at most 64")
    parser.add_argument("--scales", type=float, nargs="+", default=[.01, 1., 100.])
    parser.add_argument("--replay-protocol", type=Path, help="replay an unchanged protocol with identical source")
    args = parser.parse_args(argv)
    if args.replay_protocol is None:
        protocol = build_protocol(seed=args.seed, sizes=args.sizes, scales=args.scales)
    else:
        protocol = json.loads(args.replay_protocol.read_text(encoding="utf-8"))
    verify_protocol(protocol)
    args.output.mkdir(parents=True, exist_ok=False)
    # Freeze the complete inputs before checking native availability or solving.
    _write_new(args.output / "protocol.json", protocol)
    rows = run_cases(protocol["cases"])
    evaluation = evaluate_report(protocol, rows)
    payload = _seal({"schema": RESULT_SCHEMA, "protocol_sha256": protocol["integrity_sha256"],
        "source": _source(), "environment": environment(), "classification": "development_stress_challenge",
        "timing_scope": "call_wall_s includes canonical construction and execute; native stage times are adapter observations; no hard timeout or speed claim",
        "oracle_scope": "synthetic separable two-variable block objectives/feasibility/termination, independent of compiler and numerical kernel",
        "rows": rows, "evaluation": evaluation})
    _write_new(args.output / "results.json", payload)
    print(json.dumps({"output": str(args.output.resolve()), "passed": evaluation["passed"],
                      "case_count": len(rows), "classification": payload["classification"]}))
    return 0 if evaluation["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
