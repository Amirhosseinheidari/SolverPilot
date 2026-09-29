"""Fresh local LP selector experiment. Synthetic evidence never enables production.

Run once per output directory. Freeze this runner/protocol before timing; validation
failure stops before any test solve. Repetitions are serial and order-balanced.
"""
from dataclasses import asdict
from datetime import datetime, timezone
import argparse
import hashlib
import json
from pathlib import Path
import platform
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, solve
from solverpilot.backends import HighspyNativeBackend, PDLPBackend
from solverpilot.benchmark.environment import capture_environment, benchmark_environment_fingerprint
import solverpilot.experimental.learned_lp as learned
from solverpilot.experimental.learned_lp import (
    LPObservation, lp_features, fit_lp_selector, decide_lp_backend, evaluate_lp_selector,
)


CANDIDATES = ("highs-simplex", "highs-ipm", "ortools-pdlp")
CUTOFF = 2.0
REPEATS = 2


def backend(candidate):
    if candidate == "ortools-pdlp":
        return PDLPBackend(time_limit_s=CUTOFF, threads=1)
    return HighspyNativeBackend(time_limit_s=CUTOFF, threads=1,
                               solver="simplex" if candidate == "highs-simplex" else "ipm")


def make_problem(family, group, variant):
    seed = 31415900 + family * 10000 + group * 100 + variant
    rng = np.random.default_rng(seed)
    n = (256, 1024, 4096)[(group + variant) % 3]
    if family == 0:
        a = sparse.random(n // 2, n, density=8 / n, random_state=rng,
                          data_rvs=lambda size: rng.uniform(-1, 1, size), format="csr")
        x = rng.uniform(.1, .9, n)
        y = rng.uniform(.2, 1, n // 2) * rng.choice([-1., 1.], n // 2)
        act = a @ x
        lower, upper = np.where(y < 0, act, act - 1), np.where(y > 0, act, act + 1)
        c = np.asarray(-a.T @ y)
        reference = float(c @ x)
    elif family == 1:
        a = sparse.eye(n, format="csr")
        lower, upper = rng.uniform(.05, .25, n), rng.uniform(.75, .95, n)
        c = rng.uniform(-1, 1, n)
        reference = float(c @ np.where(c >= 0, lower, upper))
    else:
        k = int(np.sqrt(n))
        n = k * k
        cols = np.arange(n)
        a = sparse.coo_matrix((np.ones(2 * n),
                              (np.r_[cols // k, k + cols % k], np.r_[cols, cols])),
                             shape=(2 * k, n)).tocsr()
        lower = upper = np.asarray(a @ rng.uniform(.1, .9, n))
        c = rng.uniform(.1, 1., n)
        reference = None
    return LinearProblem.from_data(A=a, c=c, variable_lower=np.zeros(n),
                                   variable_upper=np.ones(n), constraint_lower=lower,
                                   constraint_upper=upper), reference


def write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    environment = capture_environment(packages=("numpy", "scipy", "highspy", "ortools"))
    # Source, versions, hardware, settings and thread policy bind the observations.
    env_id = benchmark_environment_fingerprint(environment, thread_env_limit=1,
                                              solver_threads=1, worker_python_mode="normal")
    protocol = {
        "schema": "solverpilot.local-learned-lp.v1", "captured_at": datetime.now(timezone.utc).isoformat(),
        "environment": environment, "environment_id": env_id, "platform": platform.platform(),
        "candidates": CANDIDATES, "cutoff_s": CUTOFF, "repeats": REPEATS,
        "selection": "training-only cost-sensitive stump; min_leaf_groups=4; margin_s=.005",
        "gates": ">=6 groups; >=2 switches; ratio<=.97; group-bootstrap upper<1; p90<=1.25; no lost verified solves",
        "bootstrap": {"draws": 2000, "seed": 314159},
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "selector_source_sha256": hashlib.sha256(Path(learned.__file__).read_bytes()).hexdigest(),
        "cpu_only": True, "production_authorized": False,
        "scope": "three fresh synthetic LP families; new seeds; not public/OOD or deployment qualification",
        "split_rule": "within each family groups 0..5 train, 6..7 validation, 8..9 sealed test; both variants together",
        "timing": "warm package imports; fresh solver model per call; PDLP child is cold; solve+independent validation wall time",
        "cases": [],
    }
    cases = []
    for family in range(3):
        for group in range(10):
            split = "train" if group < 6 else "validation" if group < 8 else "test"
            for variant in range(2):
                p, reference = make_problem(family, group, variant)
                rec = {"instance": f"f{family}-g{group}-v{variant}", "group": f"f{family}-g{group}",
                       "family": family, "group_index": group, "variant": variant,
                       "split": split, "data_hash": p.data_hash}
                protocol["cases"].append(rec)
                cases.append((rec, p, reference))
    if len({p.data_hash for _, p, _ in cases}) != len(cases):
        raise ValueError("duplicate generated data")
    write(args.output / "protocol.json", protocol)
    protocol_hash = hashlib.sha256((args.output / "protocol.json").read_bytes()).hexdigest()
    (args.output / "protocol.sha256").write_text(protocol_hash, encoding="ascii")
    # Fixed warm-up, outside training. Outcomes cannot change protocol or candidates.
    warm, _ = make_problem(1, 101, 0)
    warmups = []
    for name in CANDIDATES:
        result = solve(warm, backend=backend(name))
        warmups.append({"candidate": name, "verified": result.optimality_evidence.independently_verified_optimal})
    write(args.output / "warmup.json", warmups)
    all_rows = []

    def collect(split, model=None):
        observations, decisions = [], {}
        for ordinal, (rec, p, reference) in enumerate(cases):
            if rec["split"] != split:
                continue
            if model is not None:
                decisions[rec["instance"]] = decide_lp_backend(p, model, environment_id=env_id,
                                                               available=CANDIDATES)
                # Freeze the measured decision before any outcomes for this instance.
                write(args.output / f"{split}-decisions.json", {k: asdict(v) for k, v in decisions.items()})
            samples = {name: [] for name in CANDIDATES}
            for repeat in range(REPEATS):
                order = list(CANDIDATES)
                shift = ordinal % len(order)
                order = order[shift:] + order[:shift]
                if repeat % 2:
                    order.reverse()
                for name in order:
                    start = perf_counter()
                    row = {**rec, "candidate": name, "repeat": repeat}
                    try:
                        result = solve(p, backend=backend(name))
                        wall = perf_counter() - start
                        verified = result.optimality_evidence.independently_verified_optimal
                        match = reference is None or (result.objective is not None and
                            abs(result.objective - reference) <= 1e-6 * max(1, abs(reference)))
                        row.update(wall_s=wall, verified=bool(verified and match),
                                   objective=result.objective, reference=reference,
                                   reference_match=bool(match), backend_status=result.backend_status)
                    except Exception as exc:
                        row.update(wall_s=perf_counter() - start, verified=False,
                                   error=f"{type(exc).__name__}: {exc}")
                    all_rows.append(row)
                    samples[name].append((row["wall_s"], row["verified"]))
                    write(args.output / "observations.json", all_rows)
            # Independent certificates plus agreement among certified candidates.
            verified_objectives = [r["objective"] for r in all_rows
                                   if r["instance"] == rec["instance"] and r["verified"]]
            if verified_objectives:
                ref = verified_objectives[0]
                if any(abs(v - ref) > 1e-6 * max(1, abs(ref)) for v in verified_objectives):
                    raise RuntimeError("independently verified objectives disagree; campaign stopped")
            observations.append(LPObservation(rec["instance"], rec["group"], p.data_hash,
                                               split, env_id, lp_features(p), samples))
            print(json.dumps({"instance": rec["instance"], "split": split,
                              "verified": {k: sum(ok for _, ok in v) for k, v in samples.items()}}), flush=True)
        write(args.output / f"{split}.json", [asdict(r) for r in observations])
        return observations, decisions

    train, _ = collect("train")
    model = fit_lp_selector(train, candidates=CANDIDATES, cutoff_s=CUTOFF, protocol_sha256=protocol_hash)
    model.save(args.output / "model.json")
    validation, choices = collect("validation", model)
    dev = evaluate_lp_selector(model, validation, decisions=choices)
    write(args.output / "validation-report.json", dev)
    summary = {"validation": dev, "test_consumed": False, "production_authorized": False,
               "observed_solves": len(all_rows)}
    if dev["research_gate_passed"]:
        test, choices = collect("test", model)
        heldout = evaluate_lp_selector(model, test, decisions=choices)
        write(args.output / "test-report.json", heldout)
        summary.update(test=heldout, test_consumed=True, observed_solves=len(all_rows))
    else:
        summary["reason"] = "validation gate failed; test outcomes remain unobserved; no retuning on this validation cohort"
    write(args.output / "summary.json", summary)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
