from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, execute, solve
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.evaluation import performance_profile, portfolio_metrics
from solverpilot.validate import PublicStatus

OUT = Path(__file__).resolve().parent / "results" / "m2-lp-portfolio.json"

FAMILIES = {
    # name: (n_variables, n_constraints, density, equality_fraction)
    "sparse_tall": (120, 700, 0.012, 0.02),
    "sparse_wide": (700, 120, 0.012, 0.02),
    "balanced_sparse": (320, 320, 0.02, 0.05),
    "balanced_denser": (220, 220, 0.08, 0.05),
    "equality_heavy": (240, 260, 0.035, 0.35),
    "mostly_box": (900, 35, 0.02, 0.00),
}


def make_feasible_lp(seed: int, family: str) -> LinearProblem:
    n, m, density, equality_fraction = FAMILIES[family]
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=density,
        random_state=rng,
        data_rvs=lambda k: rng.normal(size=k),
        format="csr",
    )
    # Make sure every row has at least one coefficient; otherwise equality rows can become 0=0.
    A = A.tolil()
    for i in range(m):
        if len(A.rows[i]) == 0:
            j = int(rng.integers(0, n))
            A[i, j] = float(rng.normal() or 1.0)
    A = A.tocsr()

    x0 = rng.uniform(0.15, 0.85, size=n)
    ax0 = np.asarray(A @ x0).ravel()
    scale = np.maximum(0.1, np.asarray(np.abs(A).sum(axis=1)).ravel() / max(n, 1))
    slack = rng.uniform(0.1, 0.8, size=m) * scale + 0.05
    lower = ax0 - slack
    upper = ax0 + slack
    eq_count = int(round(m * equality_fraction))
    if eq_count:
        lower[:eq_count] = ax0[:eq_count]
        upper[:eq_count] = ax0[:eq_count]

    # Bounded variables prevent unbounded objective directions and make validation stable.
    c = rng.normal(size=n)
    return LinearProblem.from_data(
        A=A,
        c=c,
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lower,
        constraint_upper=upper,
        metadata={"synthetic_family": family, "seed": seed},
    )


def median_direct(problem: LinearProblem, method: str, repeats: int, order_seed: int):
    times = []
    objectives = []
    iterations = []
    backend = ScipyHighsLPBackend(method=method)
    for _ in range(repeats):
        r = execute(problem, backend)
        if r.status is not PublicStatus.VALID_OPTIMAL:
            raise AssertionError(f"{method} failed with {r.status}")
        times.append(float(r.trace.timings.total_s))
        objectives.append(float(r.objective))
        iterations.append(int((r.raw_statistics or {}).get("nit", 0)))
    return {
        "cost_s": median(times),
        "objective": median(objectives),
        "iterations_median": median(iterations),
        "samples_s": times,
    }


def main():
    # A warm-up avoids charging one-time SciPy/HiGHS initialization to the first instance.
    warm = make_feasible_lp(900001, "balanced_sparse")
    execute(warm, ScipyHighsLPBackend("highs-ds"))
    execute(warm, ScipyHighsLPBackend("highs-ipm"))

    rows = []
    repeats = 3
    t0 = perf_counter()
    for family_idx, family in enumerate(FAMILIES):
        for local in range(8):
            seed = 41000 + family_idx * 100 + local
            p = make_feasible_lp(seed, family)

            # Alternate direct-solver order to reduce systematic cache/order bias.
            methods = ["highs-ds", "highs-ipm"]
            if (seed % 2) == 1:
                methods.reverse()
            direct = {}
            for method in methods:
                direct[method] = median_direct(p, method, repeats, seed)

            auto_samples = []
            auto_selected = []
            auto_overhead = []
            for _ in range(repeats):
                r = solve(p)
                if r.status is not PublicStatus.VALID_OPTIMAL:
                    raise AssertionError(f"auto failed with {r.status}")
                auto_samples.append(float(r.trace.timings.total_s))
                auto_selected.append(r.plan.selected_backend)
                auto_overhead.append(float(r.trace.timings.inspect_s + r.trace.timings.plan_s))

            ds_obj = direct["highs-ds"]["objective"]
            ipm_obj = direct["highs-ipm"]["objective"]
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "n": p.n_variables,
                    "m": p.n_constraints,
                    "nnz": int(p.A.nnz),
                    "density": float(p.A.nnz / (p.n_variables * p.n_constraints)),
                    "objective_abs_diff": abs(ds_obj - ipm_obj),
                    "ds": direct["highs-ds"],
                    "ipm": direct["highs-ipm"],
                    "auto": {
                        "cost_s": median(auto_samples),
                        "samples_s": auto_samples,
                        "selected": max(set(auto_selected), key=auto_selected.count),
                        "overhead_s_median": median(auto_overhead),
                    },
                }
            )

    costs = {
        "scipy-highs-ds": [r["ds"]["cost_s"] for r in rows],
        "scipy-highs-ipm": [r["ipm"]["cost_s"] for r in rows],
    }
    policy_choices = [r["auto"]["selected"] for r in rows]
    policy_costs = [r["auto"]["cost_s"] for r in rows]
    metrics = portfolio_metrics(costs, policy_choices, policy_costs=policy_costs)
    taus = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0, 5.0]
    profile = performance_profile(costs, taus)

    winner_counts = {
        "scipy-highs-ds": sum(r["ds"]["cost_s"] < r["ipm"]["cost_s"] for r in rows),
        "scipy-highs-ipm": sum(r["ipm"]["cost_s"] < r["ds"]["cost_s"] for r in rows),
    }
    by_family = {}
    for family in FAMILIES:
        fam = [r for r in rows if r["family"] == family]
        by_family[family] = {
            "instances": len(fam),
            "ds_median_s": median(r["ds"]["cost_s"] for r in fam),
            "ipm_median_s": median(r["ipm"]["cost_s"] for r in fam),
            "ds_wins": sum(r["ds"]["cost_s"] < r["ipm"]["cost_s"] for r in fam),
            "ipm_wins": sum(r["ipm"]["cost_s"] < r["ds"]["cost_s"] for r in fam),
        }

    payload = {
        "benchmark": "M2 synthetic LP portfolio; development evidence only, not a market claim",
        "repeats_per_instance": repeats,
        "instances": len(rows),
        "families": FAMILIES,
        "metrics": asdict(metrics),
        "performance_profile": {"taus": taus, "fractions": profile},
        "winner_counts": winner_counts,
        "by_family": by_family,
        "max_objective_abs_diff": max(r["objective_abs_diff"] for r in rows),
        "auto_all_same_choice": len(set(policy_choices)) == 1,
        "auto_choice_set": sorted(set(policy_choices)),
        "rows": rows,
        "wall_s": perf_counter() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["instances", "metrics", "winner_counts", "by_family", "max_objective_abs_diff", "auto_choice_set", "wall_s"]}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
