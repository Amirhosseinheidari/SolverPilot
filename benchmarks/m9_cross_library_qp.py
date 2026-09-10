from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from statistics import median
from time import perf_counter

import nlopt
import numpy as np
import scipy

from solverpilot import QuadraticProblem
from solverpilot.backends import NLoptNativeBackend, ScipySLSQPQPBackend
from solverpilot.runtime import execute, execute_portfolio
from solverpilot.validate import PublicStatus

OUT = Path(__file__).parent / "results" / "m9-cross-library-qp.json"
SIZES = (5, 20, 50)
FAMILIES = ("interior", "equality", "upper_active", "box_active")
SEEDS = 4
ROUNDS = 3


def _spd(rng: np.random.Generator, n: int) -> np.ndarray:
    M = rng.normal(size=(n, n)) / np.sqrt(n)
    return M.T @ M + np.eye(n) * 0.5


def make_case(n: int, family: str, seed: int) -> tuple[QuadraticProblem, np.ndarray, float]:
    rng = np.random.default_rng(seed)
    P = _spd(rng, n)
    x_star = rng.uniform(-0.5, 0.5, size=n)
    lower = np.full(n, -2.0)
    upper = np.full(n, 2.0)
    A = np.zeros((0, n))
    cl = np.zeros(0)
    cu = np.zeros(0)

    if family == "interior":
        q = -(P @ x_star)
    elif family == "equality":
        m = min(3, max(1, n // 5))
        A = rng.normal(size=(m, n))
        # Make equality rows reasonably independent and normalized.
        A /= np.maximum(np.linalg.norm(A, axis=1, keepdims=True), 1e-12)
        b = A @ x_star
        lam = rng.uniform(-0.8, 0.8, size=m)
        q = -(P @ x_star) - A.T @ lam
        cl = b.copy(); cu = b.copy()
    elif family == "upper_active":
        a = rng.normal(size=n)
        a /= max(np.linalg.norm(a), 1e-12)
        b = float(a @ x_star)
        lam = float(rng.uniform(0.2, 1.0))
        q = -(P @ x_star) - lam * a
        # Add two inactive inequalities so the row handling is not trivial.
        a2 = rng.normal(size=n); a2 /= max(np.linalg.norm(a2), 1e-12)
        a3 = rng.normal(size=n); a3 /= max(np.linalg.norm(a3), 1e-12)
        A = np.vstack([a, a2, a3])
        cl = np.full(3, -np.inf)
        cu = np.array([b, float(a2 @ x_star + 0.5), float(a3 @ x_star + 0.5)])
    elif family == "box_active":
        # Force the first upper bound active. KKT: grad f(x*) + mu e0 = 0.
        x_star[0] = 1.0
        upper[0] = 1.0
        mu = float(rng.uniform(0.2, 1.0))
        e0 = np.zeros(n); e0[0] = 1.0
        q = -(P @ x_star) - mu * e0
    else:
        raise ValueError(family)

    p = QuadraticProblem.from_data(
        P=P,
        A=A,
        q=q,
        variable_lower=lower,
        variable_upper=upper,
        constraint_lower=cl,
        constraint_upper=cu,
        metadata={"m9_family": family, "seed": seed, "n": n},
    )
    obj = float(0.5 * x_star @ (P @ x_star) + q @ x_star)
    return p, x_star, obj


def solve_once(problem, backend):
    t0 = perf_counter()
    r = execute(problem, backend)
    wall = perf_counter() - t0
    return r, wall


def main() -> None:
    scipy_backend = ScipySLSQPQPBackend(maxiter=2000, ftol=1e-11)
    nlopt_backend = NLoptNativeBackend(max_evals=4000, ftol_rel=1e-11, xtol_rel=1e-11, constraint_tol=1e-10)

    # Warm shared libraries outside measurements.
    warm, _, _ = make_case(5, "interior", 991)
    execute(warm, scipy_backend)
    execute(warm, nlopt_backend)

    rows = []
    started = perf_counter()
    for n in SIZES:
        for fam_idx, family in enumerate(FAMILIES):
            for local_seed in range(SEEDS):
                seed = 91000 + n * 100 + fam_idx * 10 + local_seed
                p, x_star, obj_star = make_case(n, family, seed)
                times = {"scipy": [], "nlopt": []}
                results = {}
                for round_idx in range(ROUNDS):
                    order = [("scipy", scipy_backend), ("nlopt", nlopt_backend)]
                    if (seed + round_idx) % 2:
                        order.reverse()
                    for name, backend in order:
                        r, wall = solve_once(p, backend)
                        times[name].append(wall)
                        results[name] = r

                row = {"n": n, "family": family, "seed": seed, "reference_objective": obj_star}
                for name in ("scipy", "nlopt"):
                    r = results[name]
                    xerr = None if r.x is None else float(np.max(np.abs(r.x - x_star)))
                    oerr = None if r.objective is None else float(abs(r.objective - obj_star))
                    row[f"{name}_status"] = r.status.value
                    row[f"{name}_valid"] = bool(r.validation and r.validation.valid)
                    row[f"{name}_x_inf_error"] = xerr
                    row[f"{name}_objective_abs_error"] = oerr
                    row[f"{name}_median_s"] = float(median(times[name]))
                    row[f"{name}_samples_s"] = times[name]

                # Cross-library portfolio must return a validated candidate even though
                # neither backend claims an optimality certificate.
                portfolio = execute_portfolio(p, [nlopt_backend, scipy_backend], stop_on_valid_optimal=False)
                row["portfolio_chosen_backend"] = portfolio.chosen_backend
                row["portfolio_chosen_valid"] = bool(portfolio.chosen and portfolio.chosen.validation and portfolio.chosen.validation.valid)
                row["portfolio_attempts"] = len(portfolio.attempts)
                rows.append(row)

    max_x = max(float(r[f"{name}_x_inf_error"] or 0.0) for r in rows for name in ("scipy", "nlopt"))
    max_obj = max(float(r[f"{name}_objective_abs_error"] or 0.0) for r in rows for name in ("scipy", "nlopt"))
    failures = {
        name: sum(not bool(r[f"{name}_valid"]) for r in rows)
        for name in ("scipy", "nlopt")
    }
    wins = {"scipy": 0, "nlopt": 0, "ties": 0}
    for r in rows:
        a, b = float(r["scipy_median_s"]), float(r["nlopt_median_s"])
        if abs(a - b) <= 1e-9:
            wins["ties"] += 1
        elif a < b:
            wins["scipy"] += 1
        else:
            wins["nlopt"] += 1

    payload = {
        "benchmark": "M9 cross-library convex-QP correctness/timing: SciPy SLSQP vs NLopt SLSQP",
        "instances": len(rows),
        "rounds_per_backend_instance": ROUNDS,
        "solver_calls": len(rows) * ROUNDS * 2,
        "portfolio_executions": len(rows),
        "portfolio_backend_attempts": len(rows) * 2,
        "failures": failures,
        "max_x_inf_error_vs_constructed_kkt_solution": max_x,
        "max_objective_abs_error_vs_constructed_kkt_solution": max_obj,
        "timing_wins": wins,
        "portfolio_valid_count": sum(bool(r["portfolio_chosen_valid"]) for r in rows),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "nlopt": nlopt.__version__,
            "platform": platform.platform(),
            "cpu_count": os.cpu_count(),
        },
        "wall_s": perf_counter() - started,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "instances", "solver_calls", "failures",
        "max_x_inf_error_vs_constructed_kkt_solution",
        "max_objective_abs_error_vs_constructed_kkt_solution",
        "timing_wins", "portfolio_valid_count", "wall_s")}, indent=2))


if __name__ == "__main__":
    main()
