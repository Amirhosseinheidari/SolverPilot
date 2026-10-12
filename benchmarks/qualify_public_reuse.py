"""Freeze a fresh Netlib admission set, then qualify native LP reuse separately.

Preparation downloads/parses only. Run never replaces an admitted case. Published
Netlib values are historical numerical references, not exact certificates. The
four dyadic affine variants are related observations, not independent instances.
Third-party model bytes and execution artifacts stay in the explicit output dirs.
"""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
from collections import Counter
from dataclasses import asdict
from hashlib import sha256
from importlib.metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from prepare_public_lp import crosscheck_reader, fetch
from public_lp_families import family

from solverpilot import LinearProblem, read_mps
from solverpilot.backends import HighspyNativeBackend
from solverpilot.runtime.evidence import evidence_bundle, verify_evidence_bundle
from solverpilot.runtime.manifest import json_value

SCHEMA = "solverpilot.public-reuse-qualification.v1"
CAPS = {"n": 3000, "m": 2000, "nnz": 30000}
STEPS = ("baseline", "objective_scale", "rhs_and_bounds_scale", "bounds_and_rhs_translate")
TOL = {"primal_atol": 1e-7, "primal_rtol": 1e-8, "objective_atol": 1e-7,
       "objective_rtol": 1e-8, "reference_atol": 1e-5, "reference_rtol": 1e-5}
HASH_KEYS = {"raw_sha256", "mps_sha256", "data_hash", "model_hash", "input_sha256",
             "problem_data_hash", "source_hash"}


def sha(data):
    return sha256(data).hexdigest()


def write_json(path, value):
    path.write_text(json.dumps(json_value(value), indent=2, allow_nan=False) + "\n", encoding="utf-8")


def normalize_name(name):
    name = str(name).lower()
    for suffix in (".gz", ".mps", ".mat"):
        name = name.removesuffix(suffix)
    name = name.removesuffix(".netlib")
    return name.removeprefix("netlib-").removeprefix("miplib-")


def conservative_family(name):
    name = normalize_name(name)
    if name in {"gfrd-pnc", "sierra"}:
        return "forest-helgason"
    if re.match(r"^(sc205|scagr|scfxm|scorpion|scrs|scsd|sctap)", name):
        return "ho-loute-staircase"
    if name in {"25fv47", "bp", "bp1"}:
        return "25fv47"
    if name in {"fffff800", "powell"}:
        return "powell"
    if name.startswith("maros"):
        return "maros-conservative"
    return family(name)


def metadata_cases(text):
    cases = []
    pattern = (r"^([A-Z0-9][A-Z0-9.\-]*)\s+(\d+)\s+(\d+)\s+(\d+)\s+"
               r"(\d+)\s+[BR ]*\s+([+-]?[0-9.]+[ED][+-]\d+)")
    for line in text.splitlines():
        match = re.match(pattern, line)
        if match:
            name, m, n, nnz, _, reference = match.groups()
            name = name.lower()
            cases.append({"name": name, "family": conservative_family(name),
                          "n": int(n), "m": int(m) - 1, "nnz": int(nnz),
                          "reference": float(reference.replace("D", "E"))})
    if not cases or len({c["name"] for c in cases}) != len(cases):
        raise ValueError("missing or duplicate Netlib metadata records")
    return sorted(cases, key=lambda c: sha(("netlib:" + c["name"]).encode()))


def prior_from_records(records):
    """Read consumed records, not all never-downloaded candidate exclusions."""
    names, hashes = set(), set()

    def visit(value):
        if isinstance(value, list):
            for item in value:
                visit(item)
        elif isinstance(value, dict):
            if isinstance(value.get("names"), list):
                names.update(normalize_name(n) for n in value["names"])
            for key in ("name", "instance", "problem_name"):
                if isinstance(value.get(key), str):
                    names.add(normalize_name(value[key]))
            for key in HASH_KEYS:
                item = value.get(key)
                if isinstance(item, str) and re.fullmatch(r"[a-fA-F0-9]{64}", item):
                    hashes.add(item.lower())
            for key, item in value.items():
                if key == "exclusions":
                    visit([r for r in item if r.get("reason") == "pre_solve_import_failure"])
                elif key not in {"candidates", "metadata"}:
                    visit(item)

    visit(records)
    return {"names": sorted(names), "families": sorted({conservative_family(n) for n in names}),
            "hashes": sorted(hashes)}


def discover_prior(workspace):
    """Explicit known admission registers; the frozen report records their scope."""
    files = []
    for folder in ("public-lp-stage4-work", "robust-lp-stage5-metadata", "ordered-public-metadata"):
        files.append(workspace / folder / "prior-audit.json")
    model_dirs = [workspace / name for name in
                  ("public-lp-stage4-corpus", "robust-lp-stage5-corpus", "ordered-public-corpus")]
    files.extend(d / "cohort.json" for d in model_dirs)
    # Include later qualification cohorts as consumed if preparation is rerun.
    for path in workspace.glob("*/cohort.json"):
        files.append(path)
        model_dirs.append(path.parent)
    files.append(workspace / "case-study-r1-20260930/design/INPUT_ROLE_REGISTER.json")
    files.append(workspace / "routing-v2-development-corpus/manifest.json")
    repo = Path(__file__).resolve().parents[1]
    files.extend((repo / "benchmarks").glob("public-corpus*.json"))
    files.extend((repo / "docs/evidence").rglob("*cohort*.json"))
    return sorted(set(files)), model_dirs


def collect_prior(paths, model_dirs):
    records, sources = [], []
    for path in sorted(set(paths)):
        raw = path.read_bytes()  # A missing required register is a preparation error.
        records.append(json.loads(raw))
        sources.append({"path": str(path.resolve()), "sha256": sha(raw)})
    prior = prior_from_records(records)
    names, hashes = set(prior["names"]), set(prior["hashes"])
    downloaded = []
    for directory in sorted(set(model_dirs)):
        if not directory.is_dir():
            raise ValueError(f"missing consumed-model directory: {directory}")
        for path in sorted(set(directory.rglob("*.mps")) | set(directory.rglob("*.netlib"))):
            digest = sha(path.read_bytes())
            names.add(normalize_name(path.name))
            hashes.add(digest)
            downloaded.append({"path": str(path.resolve()), "sha256": digest})
    return {"names": sorted(names), "families": sorted({conservative_family(n) for n in names}),
            "hashes": sorted(hashes), "registers": sources, "downloaded_models": downloaded,
            "scope": "all named registers and downloaded models listed here; aliases are conservative, not exhaustive semantic independence"}


def admit(cases, prior, target, loader):
    selected, exclusions = [], []
    names, families, hashes = (set(prior[k]) for k in ("names", "families", "hashes"))
    used = set()
    for case in cases:
        reason = None
        if case["name"] in names:
            reason = "prior_name"
        elif case["family"] in families:
            reason = "prior_family"
        elif any(case[k] > cap for k, cap in CAPS.items()):
            reason = "structural_cap"
        elif case["family"] in used:
            reason = "selected_family"
        elif len(selected) >= target:
            reason = "quota_filled_before_download"
        if reason:
            exclusions.append({**case, "reason": reason})
            continue
        try:
            extra = loader(case)
            if any(extra[k] > cap for k, cap in CAPS.items()):
                raise ValueError("parsed structural cap")
            fresh = {extra[k] for k in ("raw_sha256", "mps_sha256", "data_hash")}
            if fresh & hashes:
                raise ValueError("previous or within-cohort raw/MPS/canonical hash")
            selected.append({**case, **extra})
            used.add(case["family"])
            hashes.update(fresh)
        except Exception as exc:  # noqa: BLE001 -- Every admission failure is an explicit observation.
            exclusions.append({**case, "reason": "pre_solve_import_failure",
                               "error": f"{type(exc).__name__}: {exc}"})
    return selected, exclusions


def prepare(args):
    paths, model_dirs = discover_prior(args.workspace)
    prior = collect_prior(paths + args.prior_json, model_dirs + args.consumed_model_dir)
    metadata = (args.metadata / "netlib-readme").read_bytes()
    decoder_source = (args.metadata / "emps.c").read_bytes()
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "netlib-readme").write_bytes(metadata)
    (args.output / "emps.c").write_bytes(decoder_source)
    decoder = json.loads(args.decoder_command)
    if not isinstance(decoder, list) or not decoder or not all(isinstance(x, str) for x in decoder):
        raise ValueError("decoder-command must be a nonempty JSON string array")

    def loader(case):
        name = case["name"]
        url = "https://www.netlib.org/lp/data/" + name
        raw = fetch(url)
        (args.output / (name + ".netlib")).write_bytes(raw)
        mps = subprocess.run(decoder, input=raw, capture_output=True, timeout=20,
                             check=True).stdout
        if not mps or len(mps) > 8_000_000:
            raise ValueError("empty or over-cap decoded MPS")
        path = args.output / (name + ".mps")
        path.write_bytes(mps)
        problem = read_mps(path)
        crosscheck_reader(path, problem)
        if problem.objective_sense.value != "minimize":
            raise ValueError("Netlib README assumes minimization")
        if problem.has_integer_variables:
            raise ValueError("continuous Netlib LP required")
        for step in STEPS:
            transformed_problem(problem, step)  # Validate all transforms before admission, no solve.
        return {"file": path.name, "url": url, "raw_sha256": sha(raw), "mps_sha256": sha(mps),
                "data_hash": problem.data_hash, "n": problem.n_variables,
                "m": problem.n_constraints, "nnz": problem.A.nnz,
                "reader_agreement": "exact arrays, bounds, sense and offset via native HiGHS MPS reader"}

    selected, exclusions = admit(metadata_cases(metadata.decode()), prior, args.target, loader)
    protocol = {"schema": SCHEMA, "target": args.target, "caps": CAPS, "tolerances": TOL,
        "selection": "SHA256(netlib:name), first importable distinct conservative families after exclusions; no solve outcomes",
        "metadata_sha256": sha(metadata), "decoder_source_sha256": sha(decoder_source),
        "decoder_command": decoder, "runner_sha256": sha(Path(__file__).read_bytes()),
        "steps": list(STEPS), "modes": ["cold", "reuse"], "presolve": [False, True],
        "repeats": args.repeats, "solve_limit_s": args.solve_limit_s, "max_wall_s": args.max_wall_s,
        "threads": 1, "solver": "simplex", "warmups": 0,
        "transform": "y=t*x+s; c'=alpha*c; bounds'=t*b+s or t*b+A*s; offset'=alpha*t*offset-alpha*c@s; reference'=alpha*t*reference",
        "timing_scope": "descriptive paired sequences, imports/parsing excluded; model build/backend construction/execute and independent check included; export/close separate; no warmup; first steps retained",
        "budget_scope": "native per-call time limit and cooperative campaign budget, not hard process cancellation",
        "scope": "fresh within frozen prior-register scope; related affine variants, historical numerical references, shared HiGHS kernel; no independent solver, exact proof, industrial, learned-routing or population-speed claim"}
    cohort = {"schema": SCHEMA, "frozen_before_solving": True, "prior": prior,
              "selected": selected, "exclusions": exclusions,
              "downloads": [{"file": p.name, "sha256": sha(p.read_bytes())}
                            for p in sorted(args.output.iterdir()) if p.suffix in {".mps", ".netlib"}],
              "missing_admissions": max(0, args.target - len(selected)),
              "complete": len(selected) == args.target}
    write_json(args.output / "protocol.json", protocol)
    write_json(args.output / "cohort.json", cohort)
    write_json(args.output / "freeze.json", {"schema": SCHEMA, "files": {
        name: sha((args.output / name).read_bytes()) for name in ("protocol.json", "cohort.json")}})
    print(json.dumps({"prepared": len(selected), "missing_admissions": cohort["missing_admissions"],
                      "names": [r["name"] for r in selected], "directory": str(args.output)}))
    return 0 if cohort["complete"] else 2


def transform_parameters(n, step):
    if step not in STEPS:
        raise ValueError("unknown frozen step")
    alpha = 1.0 if step == "baseline" else 2.0
    scale = 2.0 if step in STEPS[2:] else 1.0
    shift = ((np.arange(n) % 3) - 1) / 8 if step == STEPS[3] else np.zeros(n)
    return alpha, scale, shift


def transformed_problem(raw, step):
    alpha, scale, shift = transform_parameters(raw.n_variables, step)
    delta = np.asarray(raw.A @ shift)
    return LinearProblem.from_data(A=raw.A, c=alpha * raw.c,
        variable_lower=scale * raw.variable_lower + shift,
        variable_upper=scale * raw.variable_upper + shift,
        constraint_lower=scale * raw.constraint_lower + delta,
        constraint_upper=scale * raw.constraint_upper + delta,
        objective_sense=raw.objective_sense,
        objective_offset=alpha * scale * raw.objective_offset - float(alpha * raw.c @ shift))


def check_raw(raw, step, y, objective, reference):
    """Independent inverse-map check against ORIGINAL arrays, not target model."""
    out = {"passed": False, "primal_feasible": False, "objective_consistent": False,
           "reference_agrees": False, "max_raw_primal_violation": None,
           "raw_objective": None, "raw_reference_error": None}
    if y is None or objective is None:
        return out
    y = np.asarray(y, dtype=float)
    if y.shape != raw.c.shape or not np.isfinite(y).all() or not np.isfinite(objective):
        return out
    # Deliberately independent of transform_parameters/transformed_problem.
    if step not in STEPS:
        raise ValueError("unknown frozen step")
    translation = np.array([(i % 3 - 1) * 0.125 for i in range(len(y))]) if step == STEPS[3] else 0
    divisor = 2 if step in STEPS[2:] else 1
    multiplier = 1 if step == "baseline" else 2
    x = (y - translation) / divisor
    with np.errstate(over="ignore", invalid="ignore"):
        activity, value = raw.A @ x, float(raw.c @ x + raw.objective_offset)
        mapped = value * divisor * multiplier
    if not np.isfinite(activity).all() or not np.isfinite([value, mapped]).all():
        return out
    feasible, worst = True, 0.0
    for values, bounds, direction in ((x, raw.variable_lower, -1), (x, raw.variable_upper, 1),
            (activity, raw.constraint_lower, -1), (activity, raw.constraint_upper, 1)):
        finite = np.isfinite(bounds)
        violation = direction * (values[finite] - bounds[finite])
        worst = max(worst, float(np.max(violation, initial=0)))
        feasible = feasible and bool(np.all(violation <= TOL["primal_atol"] + TOL["primal_rtol"] * np.abs(bounds[finite])))
    consistent = abs(mapped - objective) <= TOL["objective_atol"] + TOL["objective_rtol"] * max(abs(mapped), abs(objective))
    agrees = abs(value - reference) <= TOL["reference_atol"] + TOL["reference_rtol"] * abs(reference)
    out.update(passed=bool(feasible and consistent and agrees), primal_feasible=feasible,
               objective_consistent=bool(consistent), reference_agrees=bool(agrees),
               max_raw_primal_violation=worst, raw_objective=value,
               raw_reference_error=abs(value - reference))
    return out


def load_frozen(directory):
    freeze = json.loads((directory / "freeze.json").read_text())
    if freeze.get("schema") != SCHEMA or set(freeze.get("files", {})) != {"protocol.json", "cohort.json"}:
        raise ValueError("invalid freeze manifest")
    for name, expected in freeze["files"].items():
        if sha((directory / name).read_bytes()) != expected:
            raise ValueError("frozen admission/protocol changed")
    protocol = json.loads((directory / "protocol.json").read_text())
    cohort = json.loads((directory / "cohort.json").read_text())
    if protocol["runner_sha256"] != sha(Path(__file__).read_bytes()):
        raise ValueError("runner changed after protocol freeze; create a new preparation")
    if protocol["tolerances"] != TOL or protocol["steps"] != list(STEPS):
        raise ValueError("unsupported frozen protocol")
    cases = []
    for case in cohort["selected"]:
        if Path(case["file"]).name != case["file"]:
            raise ValueError("invalid model filename")
        path = directory / case["file"]
        if sha(path.read_bytes()) != case["mps_sha256"]:
            raise ValueError("frozen MPS changed")
        raw = read_mps(path)
        if raw.data_hash != case["data_hash"]:
            raise ValueError("frozen canonical model changed")
        cases.append((case, raw))
    return protocol, cohort, cases


def row_key(row):
    return (row["name"], row["repeat"], row["presolve"], row["mode"], row["step"])


def summarize(rows, cohort, protocol):
    expected = {(c["name"], r, p, mode, step) for c in cohort["selected"]
                for r in range(protocol["repeats"]) for p in protocol["presolve"]
                for mode in protocol["modes"] for step in protocol["steps"]}
    counts = Counter(row_key(r) for r in rows)
    missing, extra = expected - counts.keys(), counts.keys() - expected
    duplicate = sum(n - 1 for n in counts.values())
    accepted = sum(r.get("accepted") is True for r in rows)
    accepted_keys = {row_key(r) for r in rows if r.get("accepted") is True} & expected
    groups = {}
    for p in protocol["presolve"]:
        for mode in protocol["modes"]:
            group = [r for r in rows if r["presolve"] == p and r["mode"] == mode]
            good = [r["checked_s"] for r in group if r.get("accepted") and r.get("checked_s") is not None]
            groups[f"{mode}/presolve={p}"] = {"intended": len(expected) // 4,
                "observed": len(group), "accepted": sum(r.get("accepted") is True for r in group),
                "accepted_only_median_checked_s": float(np.median(good)) if good else None,
                "native_reuse_applied": sum(r.get("reuse_applied") is True for r in group)}
    return {"passed": bool(cohort["complete"] and expected and not (missing or extra or duplicate)
                            and accepted == len(expected)),
            "intended": len(expected), "observed": len(rows), "accepted": accepted,
            "failed_or_unobserved": len(expected) - len(accepted_keys),
            "missing_rows": len(missing), "extra_rows": len(extra), "duplicate_rows": duplicate,
            "missing_admissions": cohort["missing_admissions"],
            "statuses": dict(Counter(r["status"] for r in rows)), "groups": groups,
            "interpretation": "correctness gate only; accepted-only timing is descriptive and cannot hide the separately counted failures"}


def run(args):
    from solverpilot import execute

    protocol, cohort, cases = load_frozen(args.prepared)
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "input-freeze.json", json.loads((args.prepared / "freeze.json").read_text()))
    rows, teardown_errors, started = [], [], perf_counter()
    deadline = started + protocol["max_wall_s"]

    def close_backend(backend, context):
        if backend is not None:
            try:
                backend.close()
            except Exception as exc:  # noqa: BLE001 -- Preserve cleanup failure without losing later rows.
                teardown_errors.append({**context, "error": f"{type(exc).__name__}: {exc}"})

    for case, raw in cases:
        for repeat in range(protocol["repeats"]):
            for presolve in protocol["presolve"]:
                modes = protocol["modes"] if repeat % 2 == 0 else protocol["modes"][::-1]
                for mode in modes:
                    backend = None
                    try:
                        for step in protocol["steps"]:
                            row = {"name": case["name"], "family": case["family"], "repeat": repeat,
                                   "presolve": presolve, "mode": mode, "step": step,
                                   "status": "campaign_budget_unobserved", "accepted": False}
                            if perf_counter() < deadline:
                                start = perf_counter()
                                try:
                                    if backend is None or mode == "cold":
                                        backend = HighspyNativeBackend(threads=1, solver="simplex",
                                            presolve=presolve, reuse=mode == "reuse",
                                            time_limit_s=protocol["solve_limit_s"])
                                    problem = transformed_problem(raw, step)
                                    built = perf_counter()
                                    result = execute(problem, backend)
                                    solved = perf_counter()
                                    check = check_raw(raw, step, result.x, result.objective, case["reference"])
                                    checked = perf_counter()
                                    row.update(status=result.status.value, backend_status=result.backend_status,
                                        objective=result.objective, candidate=json_value(result.x), check=check,
                                        internal_valid=bool(result.validation and result.validation.valid),
                                        evidence=asdict(result.optimality_evidence),
                                        phases=asdict(result.trace.timings), reuse_applied=result.trace.reuse_applied,
                                        reuse_mode=result.trace.reuse_mode, statistics=json_value(result.raw_statistics),
                                        build_s=built-start, execute_s=solved-built, check_s=checked-solved,
                                        checked_s=checked-start, completed_before_campaign_deadline=checked <= deadline)
                                    row["accepted"] = bool(result.backend_status == "optimal" and row["internal_valid"] and check["passed"])
                                    bundle = evidence_bundle(problem, result, include_raw_statistics=True)
                                    verify_evidence_bundle(bundle)
                                    evidence_path = f"evidence-{len(rows):04d}.json"
                                    write_json(args.output / evidence_path, bundle)
                                    row["evidence_file"] = evidence_path
                                except Exception as exc:  # noqa: BLE001 -- Never discard a failed intended call.
                                    row.update(status="exception", accepted=False,
                                               error=f"{type(exc).__name__}: {exc}")
                            rows.append(row)
                            with (args.output / "rows.jsonl").open("a", encoding="utf-8") as stream:
                                stream.write(json.dumps(json_value(row), allow_nan=False) + "\n")
                            if mode == "cold":
                                close_backend(backend, {"name": case["name"], "repeat": repeat,
                                                       "presolve": presolve, "mode": mode, "step": step})
                                backend = None
                    finally:
                        close_backend(backend, {"name": case["name"], "repeat": repeat,
                                                "presolve": presolve, "mode": mode})
    report = {"schema": SCHEMA, "prepared": str(args.prepared.resolve()),
              "environment": {"python": platform.python_version(), "platform": platform.platform(),
                              "numpy": version("numpy"), "scipy": version("scipy"), "highspy": version("highspy")},
              "elapsed_s": perf_counter() - started, "protocol": protocol,
              "summary": summarize(rows, cohort, protocol), "rows": rows,
              "teardown_errors": teardown_errors}
    if teardown_errors:
        report["summary"]["passed"] = False
    write_json(args.output / "result.json", report)
    print(json.dumps(report["summary"]))
    return 0 if report["summary"]["passed"] else 2


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    prep = commands.add_parser("prepare")
    prep.add_argument("--workspace", type=Path, required=True)
    prep.add_argument("--metadata", type=Path, required=True)
    prep.add_argument("--output", type=Path, required=True)
    prep.add_argument("--decoder-command", required=True, help='JSON argv, e.g. ["wsl.exe", "/path/emps", "-b"]')
    prep.add_argument("--prior-json", type=Path, action="append", default=[])
    prep.add_argument("--consumed-model-dir", type=Path, action="append", default=[])
    prep.add_argument("--target", type=int, choices=(6, 8), default=8)
    prep.add_argument("--repeats", type=int, choices=range(1, 5), default=2)
    prep.add_argument("--solve-limit-s", type=float, default=2)
    prep.add_argument("--max-wall-s", type=float, default=300)
    execute = commands.add_parser("run")
    execute.add_argument("--prepared", type=Path, required=True)
    execute.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.command == "prepare":
        if not 0 < args.solve_limit_s <= 10 or not 0 < args.max_wall_s <= 600:
            parser.error("bounded positive budgets required")
        return prepare(args)
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
