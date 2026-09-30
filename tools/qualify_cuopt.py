"""Bounded, explicit GPU/CPU comparison; writes observations, never routing rules."""

import argparse
import hashlib
from dataclasses import asdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import platform
import statistics
import subprocess
from time import perf_counter

import numpy as np
from scipy import sparse
from solverpilot import LinearProblem, solve
from solverpilot.backends import CuOptBackend, HighspyNativeBackend, PDLPBackend
import solverpilot.backends.cuopt as cuopt_adapter


def problem(n, seed):
    """Sparse bounded LP with a planted primal/dual optimum, no duplicate rows."""
    rng = np.random.default_rng(seed)
    m = max(2, n // 2)
    a = sparse.random(
        m,
        n,
        density=min(8 / n, 0.5),
        random_state=rng,
        data_rvs=lambda size: rng.uniform(-1, 1, size),
        format="csr",
    )
    x = rng.uniform(0.1, 0.9, n)
    y = rng.uniform(0.2, 1.0, m) * rng.choice([-1.0, 1.0], m)
    activity = a @ x
    lower = np.where(y < 0, activity, activity - 1.0)
    upper = np.where(y > 0, activity, activity + 1.0)
    c = np.asarray(-a.T @ y)
    p = LinearProblem.from_data(
        A=a,
        c=c,
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lower,
        constraint_upper=upper,
    )
    return p, float(c @ x)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--sizes", type=int, nargs="+", default=[500, 5000, 20000])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--time-limit", type=float, default=30.0)
    parser.add_argument(
        "--cpu-solver", choices=["choose", "simplex", "ipm", "pdlp"], default="choose"
    )
    args = parser.parse_args()
    if not 1 <= args.repeats <= 10 or any(n < 2 or n > 100000 for n in args.sizes):
        parser.error("repeats must be 1..10 and sizes 2..100000")
    if not np.isfinite(args.time_limit) or not 0 < args.time_limit <= 120:
        parser.error("time limit must be positive and at most 120 seconds")
    report = {
        "schema": "solverpilot-cuopt-qualification-v1",
        "captured_at": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
        "python": platform.python_version(),
        "versions": {name: version(name) for name in ("cuopt-cu12", "numpy", "scipy", "highspy")},
        "automatic_routing_enabled": False,
        "settings": {
            "sizes": args.sizes,
            "repeats": args.repeats,
            "time_limit_s": args.time_limit,
            "cpu_threads": 1,
            "cpu_solver": args.cpu_solver,
            "gpu_tolerance": 1e-9,
            "gpu_precision": "float64",
        },
        "adapter_sha256": hashlib.sha256(Path(cuopt_adapter.__file__).read_bytes()).hexdigest(),
        "worker_sha256": hashlib.sha256(
            Path(cuopt_adapter.__file__).with_name("_cuopt_worker.py").read_bytes()
        ).hexdigest(),
        "scope": "bounded sparse continuous LP; isolated cold cuOpt GPU solve versus "
        + (
            "isolated CPU OR-Tools PDLP"
            if args.cpu_solver == "pdlp"
            else "CPU HiGHS " + args.cpu_solver
        ),
        "records": [],
        "summary": [],
    }
    if args.cpu_solver == "pdlp":
        report["versions"]["ortools"] = version("ortools")
    try:
        report["nvidia_smi"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError) as exc:
        report["nvidia_smi"] = type(exc).__name__
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False), encoding="utf-8")

    for n in args.sizes:
        for repetition in range(args.repeats):
            p, planted = problem(n, 264800 + n + repetition)
            backends = [
                CuOptBackend(time_limit_s=args.time_limit),
                (
                    PDLPBackend(time_limit_s=args.time_limit, threads=1)
                    if args.cpu_solver == "pdlp"
                    else HighspyNativeBackend(
                        time_limit_s=args.time_limit, threads=1, solver=args.cpu_solver
                    )
                ),
            ]
            if repetition % 2:
                backends.reverse()
            for backend in backends:
                row = dict(
                    n=n,
                    m=p.n_constraints,
                    nnz=p.A.nnz,
                    repetition=repetition,
                    backend=backend.manifest.name,
                    algorithm="PDLP" if isinstance(backend, CuOptBackend) else args.cpu_solver,
                    data_hash=p.data_hash,
                    planted_objective=planted,
                )
                start = perf_counter()
                try:
                    r = solve(p, backend=backend)
                    row.update(
                        wall_s=perf_counter() - start,
                        status=r.backend_status,
                        feasible=bool(r.validation is not None and r.validation.valid),
                        independently_verified=r.optimality_evidence.independently_verified_optimal,
                        objective=r.objective,
                        trace_timings=asdict(r.trace.timings),
                        gpu_execution_reported=r.raw_statistics.get(
                            "gpu_execution_reported", False
                        ),
                        native_solve_s=r.raw_statistics.get("native_solve_s"),
                        worker_timings=dict(r.raw_statistics.get("worker_timings") or {}),
                    )
                    row["matches_planted"] = bool(
                        r.objective is not None
                        and np.isclose(r.objective, planted, atol=1e-5, rtol=1e-6)
                    )
                except Exception as exc:
                    row.update(wall_s=perf_counter() - start, error=f"{type(exc).__name__}: {exc}")
                report["records"].append(row)
                save()
                print(json.dumps(row), flush=True)
        for name in (
            "cuopt-gpu",
            "ortools-pdlp" if args.cpu_solver == "pdlp" else "highspy-native",
        ):
            rows = [r for r in report["records"] if r["n"] == n and r["backend"] == name]
            if rows:
                qualified = all(
                    r.get("feasible")
                    and r.get("independently_verified")
                    and r.get("matches_planted")
                    for r in rows
                )
                report["summary"].append(
                    dict(
                        n=n,
                        backend=name,
                        all_passed=qualified,
                        median_wall_s=statistics.median(r["wall_s"] for r in rows),
                        verified_count=sum(
                            bool(
                                r.get("feasible")
                                and r.get("independently_verified")
                                and r.get("matches_planted")
                            )
                            for r in rows
                        ),
                        trials=len(rows),
                    )
                )
        save()
    if any(not r.get("all_passed") for r in report["summary"]):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
