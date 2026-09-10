from __future__ import annotations

import json
import platform
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

from solverpilot import LinearProblem
from solverpilot.backends import ScipyVendoredHighsDevBackend

OUT = Path(__file__).resolve().parent / "results" / "m3-highs-reoptimization.json"

# Synthetic repeated-LP families.  These are deliberately not presented as market
# benchmarks; they isolate basis-reuse mechanics under stable matrix sparsity.
SIZES = {
    "small": (80, 200, 0.03),
    "medium": (150, 400, 0.03),
    "large": (250, 700, 0.03),
}
SEEDS = (7201, 7202, 7203)
REPEATS = 3
STEPS = 10


def make_sequence(seed: int, m: int, n: int, density: float, steps: int = STEPS):
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=density,
        random_state=rng,
        data_rvs=lambda k: rng.uniform(0.2, 2.0, size=k),
        format="csr",
    )
    A = A.tolil()
    # Ensure every variable appears in at least one positive equality so the feasible
    # polytope is bounded under x >= 0.
    for j in range(n):
        rows = A[:, j].nonzero()[0]
        if len(rows) == 0:
            A[int(rng.integers(0, m)), j] = float(rng.uniform(0.2, 2.0))
    A = A.tocsr()
    A.sum_duplicates()
    A.eliminate_zeros()
    A.sort_indices()

    x_base = rng.uniform(0.1, 1.0, size=n)
    c_base = rng.normal(size=n)
    problems = []
    for step in range(steps):
        x_feasible = np.maximum(0.01, x_base * (1.0 + rng.normal(scale=0.02, size=n)))
        b = np.asarray(A @ x_feasible).ravel()
        c = c_base + rng.normal(scale=0.02, size=n)
        problems.append(
            LinearProblem.from_data(
                A=A,
                c=c,
                variable_lower=np.zeros(n),
                variable_upper=np.full(n, np.inf),
                constraint_lower=b,
                constraint_upper=b,
                metadata={"seed": seed, "step": step},
            )
        )
    return problems


def run_stateful(problems):
    backend = ScipyVendoredHighsDevBackend(
        solver="simplex",
        presolve=False,
        threads=1,
    )
    rows = []
    for p in problems:
        t0 = perf_counter()
        r = backend.solve(p)
        wall = perf_counter() - t0
        if r.backend_status != "optimal":
            raise AssertionError(r.backend_status)
        rows.append(
            {
                "wall_s": wall,
                "run_s": float(r.raw_statistics["highs_run_delta_s"]),
                "iterations": int(r.raw_statistics["simplex_iteration_count"]),
                "objective": float(r.objective_reported),
                "reuse_applied": bool(r.raw_statistics["reuse_applied"]),
                "reuse_mode": str(r.raw_statistics["reuse_mode"]),
            }
        )
    return rows


def run_cold(problems):
    rows = []
    for p in problems:
        backend = ScipyVendoredHighsDevBackend(
            solver="simplex",
            presolve=False,
            threads=1,
        )
        t0 = perf_counter()
        r = backend.solve(p)
        wall = perf_counter() - t0
        if r.backend_status != "optimal":
            raise AssertionError(r.backend_status)
        rows.append(
            {
                "wall_s": wall,
                "run_s": float(r.raw_statistics["highs_run_delta_s"]),
                "iterations": int(r.raw_statistics["simplex_iteration_count"]),
                "objective": float(r.objective_reported),
                "reuse_applied": False,
                "reuse_mode": "cold_rebuild",
            }
        )
    return rows


def totals(rows):
    return {
        "wall_s": float(sum(r["wall_s"] for r in rows)),
        "run_s": float(sum(r["run_s"] for r in rows)),
        "iterations": int(sum(r["iterations"] for r in rows)),
    }


def bootstrap_median_ci(values, *, seed: int, resamples: int = 5000):
    values = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    draws = rng.choice(values, size=(resamples, values.size), replace=True)
    meds = np.median(draws, axis=1)
    return [float(np.quantile(meds, 0.025)), float(np.quantile(meds, 0.975))]


def main():
    probe = ScipyVendoredHighsDevBackend()
    if not probe.is_available():
        raise SystemExit("SciPy vendored HiGHS private bindings unavailable")

    # One-time native/SciPy initialization is warmed outside all measured sequences.
    warm = make_sequence(7999, 30, 60, 0.08, steps=2)
    run_cold(warm)
    run_stateful(warm)

    raw_runs = []
    t_all = perf_counter()
    for size_name, (m, n, density) in SIZES.items():
        for seed in SEEDS:
            problems = make_sequence(seed, m, n, density)
            for rep in range(REPEATS):
                # Alternate execution order to reduce systematic thermal/cache bias.
                if (seed + rep) % 2:
                    cold = run_cold(problems)
                    warm_rows = run_stateful(problems)
                else:
                    warm_rows = run_stateful(problems)
                    cold = run_cold(problems)

                max_obj_diff = max(
                    abs(a["objective"] - b["objective"])
                    for a, b in zip(warm_rows, cold, strict=True)
                )
                scale = max(1.0, max(abs(r["objective"]) for r in cold))
                if max_obj_diff > 1e-7 * scale:
                    raise AssertionError((size_name, seed, rep, max_obj_diff, scale))

                warm_total = totals(warm_rows)
                cold_total = totals(cold)
                raw_runs.append(
                    {
                        "size": size_name,
                        "m": m,
                        "n": n,
                        "density": density,
                        "seed": seed,
                        "repeat": rep,
                        "stateful": warm_total,
                        "cold": cold_total,
                        "wall_ratio_stateful_over_cold": warm_total["wall_s"] / cold_total["wall_s"],
                        "run_ratio_stateful_over_cold": warm_total["run_s"] / cold_total["run_s"],
                        "iteration_ratio_stateful_over_cold": warm_total["iterations"] / cold_total["iterations"],
                        "max_objective_abs_diff": max_obj_diff,
                        "stateful_reuse_steps": sum(bool(r["reuse_applied"]) for r in warm_rows),
                        "stateful_modes": [r["reuse_mode"] for r in warm_rows],
                    }
                )

    by_size = {}
    for size_name in SIZES:
        subset = [r for r in raw_runs if r["size"] == size_name]
        wall_ratios = [r["wall_ratio_stateful_over_cold"] for r in subset]
        run_ratios = [r["run_ratio_stateful_over_cold"] for r in subset]
        iteration_ratios = [r["iteration_ratio_stateful_over_cold"] for r in subset]
        size_seed = 9300 + list(SIZES).index(size_name)
        by_size[size_name] = {
            "runs": len(subset),
            "median_wall_ratio_stateful_over_cold": median(wall_ratios),
            "median_wall_ratio_bootstrap_95pct_ci": bootstrap_median_ci(wall_ratios, seed=size_seed),
            "median_run_ratio_stateful_over_cold": median(run_ratios),
            "median_run_ratio_bootstrap_95pct_ci": bootstrap_median_ci(run_ratios, seed=size_seed + 100),
            "median_iteration_ratio_stateful_over_cold": median(iteration_ratios),
            "median_iteration_ratio_bootstrap_95pct_ci": bootstrap_median_ci(iteration_ratios, seed=size_seed + 200),
            "wall_ratio_min": min(wall_ratios),
            "wall_ratio_max": max(wall_ratios),
            "all_objectives_agree": all(r["max_objective_abs_diff"] <= 1e-7 for r in subset),
            "reuse_steps_expected": STEPS - 1,
            "all_runs_applied_reuse_on_expected_steps": all(r["stateful_reuse_steps"] == STEPS - 1 for r in subset),
        }

    from scipy.optimize._highspy import _core as highs_core

    payload = {
        "benchmark": "M3 synthetic repeated-LP basis-reuse verification; not a market/SOTA benchmark",
        "methodology": {
            "solver": "HiGHS simplex via SciPy private vendored native binding",
            "highs_version": f"{highs_core.HIGHS_VERSION_MAJOR}.{highs_core.HIGHS_VERSION_MINOR}.{highs_core.HIGHS_VERSION_PATCH}",
            "presolve": False,
            "threads": 1,
            "sequence_steps": STEPS,
            "seeds": list(SEEDS),
            "repeats_per_seed": REPEATS,
            "comparison": "persistent model + explicit prior basis vs fresh solver/model each step",
            "warmup": "performed outside measurement",
            "order": "alternated by seed/repetition",
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "machine": platform.machine(),
            "scipy": scipy.__version__,
            "numpy": np.__version__,
        },
        "sizes": SIZES,
        "by_size": by_size,
        "raw_runs": raw_runs,
        "total_wall_s": perf_counter() - t_all,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({"by_size": by_size, "total_wall_s": payload["total_wall_s"]}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
