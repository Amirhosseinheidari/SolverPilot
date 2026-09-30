"""Run a preregistered, manifest-driven LP v2 DEVELOPMENT experiment.

This sequential runner measures full API calls, not solver-only time. Native
time limits are cooperative; late returns remain explicit PAR10 failures. It
does not claim a hard process deadline, fresh public evidence, or promotion.

Generate eight different structural templates without solving anything:
    python benchmarks/qualify_lp_v2.py --development-demo PATH
Then run its frozen manifest (52 API calls with the defaults):
    python benchmarks/qualify_lp_v2.py --manifest PATH/manifest.json --output RUN

External manifests use the same schema and may contain MPS or linear-json
files. A case has name, family, path, format, raw_sha256, and data_hash. Families
are canonical, externally audited structural-family labels, not random seeds.
Optional consumed/consumed_lists contain names, families, raw_sha256, and
data_hashes arrays. Consumed identities are permitted only in training.
"""
from __future__ import annotations

import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import statistics
from time import perf_counter


SCHEMA = "solverpilot.lp-v2-development-manifest.v1"
CANDIDATES = ("highs-simplex", "highs-ipm")
BASELINES = ("production", "default")
SPLITS = ("train", "calibration", "test")
IDENTITIES = ("name", "family", "raw_sha256", "data_hash")
CONSUMED_KEYS = dict(name="names", family="families", raw_sha256="raw_sha256", data_hash="data_hashes")


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def file_sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def normalized_identity(value):
    if not isinstance(value, str) or not value.strip():
        raise ValueError("missing canonical name/family")
    return value.strip().casefold()


def checked_digest(value):
    if not isinstance(value, str) or len(value) != 64 or any(c not in "0123456789abcdef" for c in value):
        raise ValueError("invalid SHA256/data hash")
    return value


def load_problem(path, file_format):
    from solverpilot import LinearProblem, read_mps
    if file_format == "mps":
        problem = read_mps(path)
    elif file_format == "linear-json":
        import numpy as np
        from scipy import sparse
        spec = json.loads(Path(path).read_text(encoding="utf-8"))
        def bounds(name, missing):
            return [missing if x is None else x for x in spec[name]]
        matrix = spec["A"] or sparse.csr_matrix((0, len(spec["c"])))
        problem = LinearProblem.from_data(A=matrix, c=spec["c"],
            variable_lower=bounds("variable_lower", -np.inf),
            variable_upper=bounds("variable_upper", np.inf),
            constraint_lower=bounds("constraint_lower", -np.inf),
            constraint_upper=bounds("constraint_upper", np.inf),
            objective_sense=spec.get("objective_sense", "minimize"),
            objective_offset=spec.get("objective_offset", 0.))
    else:
        raise ValueError("format must be mps or linear-json")
    if problem.has_integer_variables:
        raise ValueError("manifest must contain continuous LPs; relaxation must be prepared explicitly")
    return problem


def validate_manifest(manifest, base, *, loader=load_problem, max_calls=80):
    """Check every identity, byte string, and parsed hash before any solver call."""
    if manifest.get("schema") != SCHEMA or manifest.get("scope") != "development":
        raise ValueError("only the explicit development manifest schema is supported")
    config = dict(repeats=2, cutoff_s=2., reuse_count=20, min_leaf_families=2,
                  calibration_min_families=2, max_depth=2, switch_margin_s=.005,
                  overhead_limit_s=.005)
    config.update(manifest.get("protocol", {}))
    if set(config) != {"repeats", "cutoff_s", "reuse_count", "min_leaf_families",
                       "calibration_min_families", "max_depth", "switch_margin_s", "overhead_limit_s"}:
        raise ValueError("unknown protocol option")
    for name in ("repeats", "reuse_count", "min_leaf_families", "calibration_min_families"):
        if type(config[name]) is not int or config[name] < (2 if name in {"repeats", "min_leaf_families", "calibration_min_families"} else 1):
            raise ValueError("invalid protocol " + name)
    if type(config["max_depth"]) is not int or not 0 <= config["max_depth"] <= 2:
        raise ValueError("invalid protocol max_depth")
    for name in ("cutoff_s", "switch_margin_s", "overhead_limit_s"):
        value = config[name]
        if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0 or (name == "cutoff_s" and value == 0):
            raise ValueError("invalid protocol " + name)
    if tuple(manifest.get("candidates", CANDIDATES)) != CANDIDATES:
        raise ValueError("candidate algorithms/settings are fixed by this runner")
    if set(manifest.get("splits", {})) != set(SPLITS):
        raise ValueError("explicit train/calibration/test splits required")
    consumed = {key: set() for key in IDENTITIES}
    consumption_documents = [manifest.get("consumed", {})]
    consumption_hashes = {}
    for name in manifest.get("consumed_lists", []):
        path = (Path(base)/name).resolve()
        consumption_documents.append(json.loads(path.read_text(encoding="utf-8")))
        consumption_hashes[str(path)] = file_sha(path)
    for document in consumption_documents:
        if not isinstance(document, dict) or set(document)-set(CONSUMED_KEYS.values()):
            raise ValueError("invalid consumed identity list")
        for key, field in CONSUMED_KEYS.items():
            values = document.get(field, [])
            if not isinstance(values, list):
                raise ValueError("consumed identities must be arrays")
            consumed[key].update((normalized_identity(v) if key in {"name", "family"}
                                  else checked_digest(v)) for v in values)
    seen = {key: {} for key in IDENTITIES}
    cases = {}
    for split in SPLITS:
        cases[split] = []
        raw_cases = manifest["splits"][split]
        if not isinstance(raw_cases, list) or not raw_cases:
            raise ValueError("each split must contain explicit cases")
        for original in raw_cases:
            case = dict(original)
            for key in IDENTITIES:
                value = (normalized_identity(case.get(key)) if key in {"name", "family"}
                         else checked_digest(case.get(key)))
                prior = seen[key].get(value)
                if prior is not None and (prior != split or key != "family"):
                    raise ValueError("duplicate or cross-split overlap: " + key)
                if split != "train" and value in consumed[key]:
                    raise ValueError("consumed calibration/test " + key)
                seen[key][value] = split
                case[key] = value
            if case.get("format") not in {"mps", "linear-json"} or not isinstance(case.get("path"), str):
                raise ValueError("case needs explicit path and format")
            reference = case.get("reference")
            if reference is not None and (isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference)):
                raise ValueError("reference must be finite")
            case["path"] = str((Path(base)/case["path"]).resolve())
            cases[split].append(case)
    calls = config["repeats"] * (2*len(cases["train"]) + 4*len(cases["calibration"]) + 5*len(cases["test"]))
    if type(max_calls) is not int or max_calls <= 0 or calls > max_calls:
        raise ValueError(f"planned {calls} API calls exceed explicit cap {max_calls}")
    problems = {}
    for split in SPLITS:
        for case in cases[split]:
            if file_sha(case["path"]) != case["raw_sha256"]:
                raise ValueError("corpus raw SHA256 mismatch: " + case["name"])
            problem = loader(case["path"], case["format"])
            if problem.data_hash != case["data_hash"]:
                raise ValueError("canonical data hash mismatch: " + case["name"])
            problems[case["name"]] = problem
    return config, cases, problems, consumption_hashes, calls


def source_hashes():
    root = Path(__file__).resolve().parents[1]
    paths = [Path(__file__).resolve(), *sorted((root/"src"/"solverpilot").rglob("*.py"))]
    return {str(path.relative_to(root)): file_sha(path) for path in paths}


def verify_frozen(expected_sources, artifacts):
    if source_hashes() != expected_sources:
        raise ValueError("implementation changed after protocol/model freeze")
    for path, expected in artifacts.items():
        if file_sha(path) != expected:
            raise ValueError("frozen artifact changed: " + str(path))


def strategies(split):
    return CANDIDATES if split == "train" else (*CANDIDATES, *BASELINES) if split == "calibration" else ("v2", *CANDIDATES, *BASELINES)


def strategy_order(split, case_index, repeat):
    order = list(strategies(split))
    shift = case_index % len(order)
    order = order[shift:]+order[:shift]
    # Pair each order with its exact reverse. Including repeat in the rotation
    # would cancel the reversal for two candidates and bias first-call costs.
    return order if repeat % 2 == 0 else list(reversed(order))


def validate_outcome_coverage(rows, cases, repeats, split):
    expected = {(case["name"], strategy, repeat) for case in cases
                for strategy in strategies(split) for repeat in range(repeats)}
    keys = [(row["name"], row["strategy"], row["repeat"]) for row in rows]
    if len(keys) != len(set(keys)) or set(keys) != expected:
        raise ValueError("missing, extra, or duplicate outcomes; no rows may be discarded")


def check_objectives(rows, cases):
    """Keep raw verification flags, but disqualify conflicting objective evidence."""
    for case in cases:
        selected = [row for row in rows if row["name"] == case["name"]]
        values = [row.get("objective") for row in selected if row.get("verified")]
        valid = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) for v in values)
        reference = case.get("reference", values[0] if values else None)
        if reference is None and values:
            reference = values[0]
        if valid and reference is not None:
            valid = all(abs(v-reference) <= 1e-6*max(1., abs(v), abs(reference)) for v in values)
        for row in selected:
            row["objective_consistent"] = valid


def score_outcome(row, *, cutoff_s, reuse_count, scope):
    if scope not in {"cold", "amortized"} or type(reuse_count) is not int or reuse_count < 1:
        raise ValueError("invalid cost scope/reuse count")
    wall, prep = row.get("api_wall_s"), row.get("preparation_s", 0.)
    finite = all(isinstance(v, (int, float)) and not isinstance(v, bool) and math.isfinite(v) and v >= 0 for v in (wall, prep))
    charged = wall + prep/(1 if scope == "cold" else reuse_count) if finite else None
    success = bool(row.get("verified") is True and row.get("objective_consistent", True)
                   and charged is not None and charged <= cutoff_s)
    return {"charged_s": charged, "success": success,
            "par10_s": charged if success else 10*cutoff_s}


def summarize(rows, *, cutoff_s, reuse_count):
    report = {"scope": "development_only", "automatic_production_routing_enabled": False,
              "production_promotion_supported": False, "outcome_count": len(rows),
              "reuse_count": reuse_count, "actual_v2_calls": sum(r["strategy"] == "v2" for r in rows),
              "amortization_is_preregistered_scenario": True, "cost_scopes": {}}
    for scope in ("cold", "amortized"):
        values = {}
        for strategy in sorted({row["strategy"] for row in rows}):
            selected = [row for row in rows if row["strategy"] == strategy]
            scores = [score_outcome(row, cutoff_s=cutoff_s, reuse_count=reuse_count, scope=scope) for row in selected]
            families = sorted({row["family"] for row in selected})
            family_costs = [statistics.mean(score["par10_s"] for row, score in zip(selected, scores)
                                            if row["family"] == family) for family in families]
            values[strategy] = {"outcomes": len(selected), "families": len(families),
                "verified_within_budget": sum(score["success"] for score in scores),
                "failed_or_late": sum(not score["success"] for score in scores),
                "mean_par10_s": statistics.mean(score["par10_s"] for score in scores),
                "family_mean_par10_s": statistics.mean(family_costs)}
        report["cost_scopes"][scope] = values
    return report


class NativeRuntime:
    """Small adapter so protocol/accounting tests need no native solver."""
    def __init__(self):
        from solverpilot.benchmark.environment import capture_environment, benchmark_environment_fingerprint
        self.environment = capture_environment(packages=("numpy", "scipy", "highspy"))
        self.environment_id = benchmark_environment_fingerprint(self.environment,
            thread_env_limit=1, solver_threads=1, worker_python_mode="normal")

    def features(self, problem):
        from solverpilot.experimental.learned_lp_v2 import lp_features_v2
        return lp_features_v2(problem)

    def fit(self, rows, config, protocol_hash):
        from solverpilot.experimental.learned_lp_v2 import fit_lp_selector_v2
        return fit_lp_selector_v2(rows, candidates=CANDIDATES, cutoff_s=config["cutoff_s"],
            protocol_sha256=protocol_hash, min_leaf_families=config["min_leaf_families"],
            max_depth=config["max_depth"], switch_margin_s=config["switch_margin_s"])

    def calibrate(self, model, rows, config):
        from solverpilot.experimental.learned_lp_v2 import calibrate_lp_guard_v2
        return calibrate_lp_guard_v2(model, rows, overhead_limit_s=config["overhead_limit_s"],
            setup_charge_s=0., min_families=config["calibration_min_families"])

    def prepare(self, model_path, guard_path, config):
        from solverpilot.backends import HighspyNativeBackend
        from solverpilot.experimental.learned_lp_v2 import LPSelectorV2, LPLeafGainGuardV2
        from solverpilot.experimental.lp_session import LPRoutingSession
        model, guard = LPSelectorV2.load(model_path), LPLeafGainGuardV2.load(guard_path)
        backends = {name: HighspyNativeBackend(time_limit_s=config["cutoff_s"], threads=1,
            solver="simplex" if name == "highs-simplex" else "ipm") for name in CANDIDATES}
        session = LPRoutingSession.create(model, guard, environment_id=self.environment_id,
                                          cutoff_s=config["cutoff_s"], backends=backends)
        return model, guard, session, backends

    def execute(self, problem, strategy, config, prepared):
        from solverpilot import solve, solve_production, SolveBudget
        from solverpilot.backends import HighspyNativeBackend
        from solverpilot.experimental.robust_lp import solve_robust_lp
        cutoff = config["cutoff_s"]
        budget = SolveBudget(wall_time_s=cutoff)
        if strategy == "default":
            result = solve(problem, budget=budget)
        elif strategy == "production":
            result, _ = solve_production(problem, budget=budget)
        elif strategy == "v2":
            model, guard, session, backends = prepared
            result = solve_robust_lp(problem, model, environment_id=self.environment_id,
                backends=backends, time_limit_s=cutoff, gain_guard=guard, session=session)
        else:
            backend = HighspyNativeBackend(time_limit_s=cutoff, threads=1,
                solver="simplex" if strategy == "highs-simplex" else "ipm")
            result = solve(problem, backend=backend, budget=budget)
        return {"status": result.status.value, "backend_status": result.backend_status,
            "backend": result.trace.backend, "objective": result.objective,
            "primal_valid": bool(result.validation and result.validation.valid),
            "verified": bool(result.optimality_evidence.independently_verified_optimal),
            "route": dict(result.raw_statistics.get("experimental_lp_route", {}))}


def observations(rows, cases, features, split, environment_id, config):
    from solverpilot.experimental.learned_lp import LPObservation
    selected_candidates = CANDIDATES if split == "train" else (*CANDIDATES, "production")
    return [LPObservation(case["name"], case["family"], case["data_hash"], split,
        environment_id, tuple(features[case["name"]]), {candidate: tuple(
            (row["api_wall_s"], score_outcome(row, cutoff_s=config["cutoff_s"],
                reuse_count=config["reuse_count"], scope="cold")["success"])
            for row in rows if row["name"] == case["name"] and row["strategy"] == candidate)
            for candidate in selected_candidates}) for case in cases]


def run_protocol(manifest_path, output, *, runtime=None, max_calls=80):
    # Set thread limits before importing numerical libraries in the CLI path.
    for name in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS", "BLIS_NUM_THREADS"):
        os.environ[name] = "1"
    manifest_path, output = Path(manifest_path).resolve(), Path(output).resolve()
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    config, cases, problems, consumed_hashes, calls = validate_manifest(manifest, manifest_path.parent, max_calls=max_calls)
    runtime = runtime or NativeRuntime()
    output.mkdir(parents=True, exist_ok=False)
    sources = source_hashes()
    protocol = {"schema": "solverpilot.lp-v2-development-protocol.v1", "created_at": datetime.now(timezone.utc).isoformat(),
        "scope": "development_only", "manifest_sha256": file_sha(manifest_path),
        "consumed_list_sha256": consumed_hashes, "implementation_sha256": digest(sources),
        "source_hashes": sources, "environment": runtime.environment, "environment_id": runtime.environment_id,
        "config": config, "cases": cases, "planned_api_calls": calls, "api_call_cap": max_calls,
        "candidate_settings": {name: {"backend": "highspy-native", "solver": "simplex" if name == "highs-simplex" else "ipm", "threads": 1} for name in CANDIDATES},
        "baselines": BASELINES, "parallel_solves": 1, "hard_process_deadline": False,
        "cutoff_semantics": "full API call; cooperative native limits; late/failed/unverified repeats receive 10*cutoff",
        "cost_scopes": {"cold": "measured preparation + full API", "amortized": "measured preparation / preregistered K + full API"},
        "guard_scope": "warm calls: setup_charge_s=0; no cold-performance inference from calibration",
        "preparation_scope": "frozen model/guard disk load and implementation/environment session binding; once per explicit session",
        "common_input_scope": "manifest verification and LP parsing before timing, common to all strategies",
        "family_scope": "manifest-declared structural families; no automatic independent family audit",
        "retuning_allowed": False, "test_evaluations": 1, "automatic_production_routing_enabled": False,
        "public_promotion_supported": False}
    write_json(output/"protocol.json", protocol)
    write_json(output/"manifest.json", manifest)
    frozen = {str(output/"protocol.json"): file_sha(output/"protocol.json"),
              str(output/"manifest.json"): file_sha(output/"manifest.json"),
              str(manifest_path): file_sha(manifest_path), **consumed_hashes,
              **{case["path"]: case["raw_sha256"] for split in SPLITS for case in cases[split]}}
    prepared, prep_s, preparation_error = None, 0., None
    features = {}

    def collect(split):
        rows = []
        for case_index, case in enumerate(cases[split]):
            if split != "test":
                features[case["name"]] = tuple(runtime.features(problems[case["name"]]))
            for repeat in range(config["repeats"]):
                order = strategy_order(split, case_index, repeat)
                for strategy in order:
                    verify_frozen(sources, frozen)
                    start = perf_counter()
                    try:
                        if strategy == "v2" and preparation_error:
                            raise RuntimeError(preparation_error)
                        outcome = runtime.execute(problems[case["name"]], strategy, config, prepared)
                        if not isinstance(outcome, dict):
                            raise ValueError("backend returned a malformed outcome")
                    except Exception as exc:
                        outcome = {"status": "exception", "verified": False,
                                   "error": f"{type(exc).__name__}: {exc}"}
                    elapsed = perf_counter()-start
                    row = {**outcome, "name": case["name"], "family": case["family"],
                        "data_hash": case["data_hash"], "split": split, "strategy": strategy,
                        "repeat": repeat, "api_wall_s": elapsed,
                        "preparation_s": prep_s if strategy == "v2" else 0.}
                    value = row.get("objective")
                    if value is not None and (isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value)):
                        row["objective"] = None
                        row["verified"] = False
                    rows.append(row)
                    write_json(output/(split+"-outcomes.json"), rows)
        validate_outcome_coverage(rows, cases[split], config["repeats"], split)
        check_objectives(rows, cases[split])
        write_json(output/(split+"-outcomes.json"), rows)
        return rows

    train = collect("train")
    train_rows = observations(train, cases["train"], features, "train", runtime.environment_id, config)
    write_json(output/"training-observations.json", [asdict(row) for row in train_rows])
    model = runtime.fit(train_rows, config, file_sha(output/"protocol.json"))
    model.save(output/"model.json")
    frozen[str(output/"model.json")] = file_sha(output/"model.json")
    write_json(output/"model-freeze.json", {"sha256": frozen[str(output/"model.json")],
        "before_calibration": True, "implementation_sha256": digest(sources)})
    calibration = collect("calibration")
    calibration_rows = observations(calibration, cases["calibration"], features, "calibration", runtime.environment_id, config)
    write_json(output/"calibration-observations.json", [asdict(row) for row in calibration_rows])
    guard = runtime.calibrate(model, calibration_rows, config)
    guard.save(output/"guard.json")
    frozen[str(output/"guard.json")] = file_sha(output/"guard.json")
    write_json(output/"guard-freeze.json", {"sha256": frozen[str(output/"guard.json")],
        "model_sha256": frozen[str(output/"model.json")], "before_test": True,
        "implementation_sha256": digest(sources), "retuning_allowed": False})
    verify_frozen(sources, frozen)
    start = perf_counter()
    try:
        prepared = runtime.prepare(output/"model.json", output/"guard.json", config)
    except Exception as exc:
        preparation_error = f"{type(exc).__name__}: {exc}"
    prep_s = perf_counter()-start
    write_json(output/"session-preparation.json", {"preparation_s": prep_s,
        "reuse_count": config["reuse_count"], "error": preparation_error})
    test = collect("test")
    verify_frozen(sources, frozen)
    summary = summarize(test, cutoff_s=config["cutoff_s"], reuse_count=config["reuse_count"])
    summary.update(total_api_calls=len(train)+len(calibration)+len(test),
                   implementation_sha256=digest(sources), model_sha256=frozen[str(output/"model.json")],
                   guard_sha256=frozen[str(output/"guard.json")], preparation_s=prep_s,
                   test_evaluations=1)
    write_json(output/"summary.json", summary)
    return summary


def generate_development_demo(output):
    """Eight explicit structural templates; no random-seed family inflation."""
    output = Path(output).resolve()
    output.mkdir(parents=True, exist_ok=False)
    templates = [
        ("box", [], [1., -2., .5], [0.]*3, [2.]*3, [], []),
        ("packing", [[1., 2., 1., 3.]], [-1., -2., -3., -1.], [0.]*4, [2.]*4, [None], [4.]),
        ("covering", [[1., 1., 0.], [0., 1., 1.]], [1., 2., 1.], [0.]*3, [3.]*3, [1., 2.], [None]*2),
        ("path-flow", [[1., 0., -1., 0.], [-1., 1., 0., 0.], [0., -1., 0., 1.]], [1., 1., 2., 1.], [0.]*4, [3.]*4, [0., 0., 0.], [0., 0., 0.]),
        ("transport", [[1., 1., 1., 0., 0., 0.], [0., 0., 0., 1., 1., 1.], [1., 0., 0., 1., 0., 0.], [0., 1., 0., 0., 1., 0.]], [1., 2., 3., 3., 2., 1.], [0.]*6, [3.]*6, [3., 3., 2., 2.], [3., 3., 2., 2.]),
        ("epigraph", [[1., 1., -1.], [-1., 1., -1.], [1., -1., -1.], [-1., -1., -1.]], [0., 0., 1.], [-1., -1., 0.], [1., 1., 3.], [None]*4, [0.]*4),
        ("triangular-equality", [[1., 0., 0.], [1., 1., 0.], [0., 1., 1.]], [1., 2., 3.], [None]*3, [None]*3, [1., 3., 5.], [1., 3., 5.]),
        ("ranged-band", [[1., -1., 0., 0., 0.], [0., 1., -1., 0., 0.], [0., 0., 1., -1., 0.], [0., 0., 0., 1., -1.]], [1., -1., 2., -2., .5], [0.]*5, [2.]*5, [-.5]*4, [.5]*4),
    ]
    splits = {split: [] for split in SPLITS}
    for index, (name, a, c, vl, vu, cl, cu) in enumerate(templates):
        path = output/(name+".json")
        write_json(path, dict(A=a, c=c, variable_lower=vl, variable_upper=vu,
                             constraint_lower=cl, constraint_upper=cu))
        problem = load_problem(path, "linear-json")
        split = "train" if index < 4 else "calibration" if index < 6 else "test"
        splits[split].append(dict(name="demo-"+name, family="synthetic-template:"+name,
            path=path.name, format="linear-json", raw_sha256=file_sha(path), data_hash=problem.data_hash))
    manifest = {"schema": SCHEMA, "scope": "development", "candidates": CANDIDATES,
        "description": "Eight deterministic, different structural templates. Consumed development evidence; no public/generalization promotion.",
        "protocol": {"repeats": 2, "cutoff_s": 2., "reuse_count": 20,
                     "min_leaf_families": 2, "calibration_min_families": 2}, "splits": splits}
    write_json(output/"manifest.json", manifest)
    return output/"manifest.json"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--development-demo", type=Path)
    parser.add_argument("--max-calls", type=int, default=80)
    args = parser.parse_args()
    if args.development_demo:
        if args.manifest or args.output:
            parser.error("demo generation is separate from measurement")
        print(generate_development_demo(args.development_demo))
        return
    if not args.manifest or not args.output:
        parser.error("--manifest and a new --output directory are required")
    print(json.dumps(run_protocol(args.manifest, args.output, max_calls=args.max_calls), indent=2))


if __name__ == "__main__":
    main()
