from __future__ import annotations

import json
import math
import os
from pathlib import Path
import platform
import sys
from statistics import mean, median
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

from solverpilot import LinearProblem, execute
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.evaluation import portfolio_metrics
from solverpilot.inspect import inspect_problem
from solverpilot.validate import PublicStatus

OUT = Path(__file__).parent / "results" / "m5-lp-selector-stability.json"
FAMILIES = {
    "tall_moderate": dict(n=300, m=800, density=0.025, equality_fraction=0.05),
    "tall_strong": dict(n=220, m=900, density=0.025, equality_fraction=0.05),
    "wide_moderate": dict(n=800, m=300, density=0.025, equality_fraction=0.05),
    "balanced_sparse": dict(n=500, m=500, density=0.015, equality_fraction=0.05),
    "balanced_dense": dict(n=250, m=250, density=0.25, equality_fraction=0.05),
    "equality_heavy": dict(n=400, m=400, density=0.03, equality_fraction=0.50),
}
SEEDS_PER_FAMILY = 4
ROUNDS = 5


def make_feasible_lp(seed: int, family: str) -> LinearProblem:
    cfg = FAMILIES[family]
    n, m = cfg["n"], cfg["m"]
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=cfg["density"],
        random_state=rng,
        data_rvs=lambda k: rng.normal(size=k),
        format="csr",
    ).tolil()
    for i in range(m):
        if not A.rows[i]:
            A[i, int(rng.integers(0, n))] = 1.0
    A = A.tocsr()
    x0 = rng.uniform(0.2, 0.8, size=n)
    ax0 = np.asarray(A @ x0).ravel()
    row_mass = np.asarray(np.abs(A).sum(axis=1)).ravel()
    slack = 0.05 + rng.uniform(0.05, 0.25, size=m) * np.maximum(1e-6, row_mass / max(n, 1))
    lower, upper = ax0 - slack, ax0 + slack
    eq_count = int(round(cfg["equality_fraction"] * m))
    lower[:eq_count] = ax0[:eq_count]
    upper[:eq_count] = ax0[:eq_count]
    return LinearProblem.from_data(
        A=A,
        c=rng.normal(size=n),
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lower,
        constraint_upper=upper,
        metadata={"synthetic_family": family, "seed": seed},
    )


def solve_once(problem: LinearProblem, method: str) -> tuple[float, float]:
    r = execute(problem, ScipyHighsLPBackend(method=method))
    if r.status is not PublicStatus.VALID_OPTIMAL:
        raise AssertionError(f"{method} failed: {r.status}")
    return float(r.trace.timings.total_s), float(r.objective)


def rules() -> dict[str, float | None]:
    return {
        "always_ds": None,
        "always_ipm": 0.0,
        "ipm_if_ratio_ge_1.50": 1.50,
        "ipm_if_ratio_ge_2.00": 2.00,
        "ipm_if_ratio_ge_2.50": 2.50,
        "ipm_if_ratio_ge_3.00": 3.00,
    }


def choose(threshold: float | None, row: dict[str, object]) -> str:
    if threshold is None:
        return "scipy-highs-ds"
    if threshold == 0.0:
        return "scipy-highs-ipm"
    return "scipy-highs-ipm" if float(row["m_over_n"]) >= threshold else "scipy-highs-ds"


def train_cost(threshold: float | None, rows: list[dict[str, object]]) -> float:
    vals = []
    for row in rows:
        method = choose(threshold, row)
        vals.append(float(row["ipm_median_s"] if method.endswith("ipm") else row["ds_median_s"]))
    return float(mean(vals))


def bootstrap_ratio(policy: np.ndarray, sbs: np.ndarray, draws: int = 10000) -> dict[str, float]:
    rng = np.random.default_rng(20260831)
    ratios = np.empty(draws)
    n = policy.size
    for k in range(draws):
        idx = rng.integers(0, n, size=n)
        ratios[k] = np.mean(policy[idx]) / np.mean(sbs[idx])
    q = np.quantile(ratios, [0.025, 0.5, 0.975])
    return {"point": float(np.mean(policy) / np.mean(sbs)), "low": float(q[0]), "median": float(q[1]), "high": float(q[2]), "draws": draws}


def main() -> None:
    # Warm both algorithms outside measurements.
    warm = make_feasible_lp(999001, "balanced_sparse")
    solve_once(warm, "highs-ds")
    solve_once(warm, "highs-ipm")

    started = perf_counter()
    rows: list[dict[str, object]] = []
    for family_index, family in enumerate(FAMILIES):
        for local_seed in range(SEEDS_PER_FAMILY):
            seed = 73000 + family_index * 100 + local_seed
            p = make_feasible_lp(seed, family)
            inspect_samples = []
            ds_samples, ipm_samples = [], []
            ds_obj, ipm_obj = None, None
            round_winners = []
            for round_index in range(ROUNDS):
                t0 = perf_counter(); inspect_problem(p); inspect_samples.append(perf_counter() - t0)
                order = ["highs-ds", "highs-ipm"]
                if (seed + round_index) % 2:
                    order.reverse()
                observed = {}
                for method in order:
                    t, obj = solve_once(p, method)
                    observed[method] = t
                    if method.endswith("ds"):
                        ds_samples.append(t); ds_obj = obj
                    else:
                        ipm_samples.append(t); ipm_obj = obj
                round_winners.append("ds" if observed["highs-ds"] < observed["highs-ipm"] else "ipm")
            assert ds_obj is not None and ipm_obj is not None
            diff = abs(ds_obj - ipm_obj)
            if diff > 1e-6 * max(1.0, abs(ds_obj)):
                raise AssertionError(f"objective mismatch {family}/{seed}: {diff}")
            rows.append({
                "family": family,
                "seed": seed,
                "n": p.n_variables,
                "m": p.n_constraints,
                "m_over_n": p.n_constraints / p.n_variables,
                "inspect_median_s": float(median(inspect_samples)),
                "ds_median_s": float(median(ds_samples)),
                "ipm_median_s": float(median(ipm_samples)),
                "ds_samples_s": ds_samples,
                "ipm_samples_s": ipm_samples,
                "ds_round_wins": round_winners.count("ds"),
                "ipm_round_wins": round_winners.count("ipm"),
                "median_winner": "ds" if median(ds_samples) < median(ipm_samples) else "ipm",
                "objective_abs_diff": diff,
            })

    policy_choices: dict[tuple[str, int], str] = {}
    folds = []
    for held_out in FAMILIES:
        train = [r for r in rows if r["family"] != held_out]
        test = [r for r in rows if r["family"] == held_out]
        scores = [(train_cost(th, train), name, th) for name, th in rules().items()]
        tr_cost, rule_name, threshold = min(scores, key=lambda x: (x[0], x[1]))
        for row in test:
            policy_choices[(str(row["family"]), int(row["seed"]))] = choose(threshold, row)
        folds.append({"held_out_family": held_out, "rule": rule_name, "threshold": threshold, "training_mean_s": tr_cost})

    costs = {
        "scipy-highs-ds": [float(r["ds_median_s"]) for r in rows],
        "scipy-highs-ipm": [float(r["ipm_median_s"]) for r in rows],
    }
    choices = [policy_choices[(str(r["family"]), int(r["seed"]))] for r in rows]
    policy_costs = []
    for row, method in zip(rows, choices):
        base = float(row["ipm_median_s"] if method.endswith("ipm") else row["ds_median_s"])
        policy_costs.append(base + float(row["inspect_median_s"]))
    metrics = portfolio_metrics(costs, choices, policy_costs=policy_costs)
    sbs = np.asarray(costs[metrics.sbs_solver])
    policy = np.asarray(policy_costs)

    by_family = {}
    for family in FAMILIES:
        fam = [r for r in rows if r["family"] == family]
        by_family[family] = {
            "median_winner_counts": {
                "ds": sum(r["median_winner"] == "ds" for r in fam),
                "ipm": sum(r["median_winner"] == "ipm" for r in fam),
            },
            "round_winner_counts": {
                "ds": sum(int(r["ds_round_wins"]) for r in fam),
                "ipm": sum(int(r["ipm_round_wins"]) for r in fam),
            },
        }

    stable_instances = sum(max(int(r["ds_round_wins"]), int(r["ipm_round_wins"])) >= 4 for r in rows)
    payload = {
        "benchmark": "M5 timing-stability stress test of M4 synthetic LP algorithm-selection evidence",
        "instances": len(rows),
        "rounds_per_solver_instance": ROUNDS,
        "solver_calls": len(rows) * ROUNDS * 2,
        "metrics": {
            "sbs_solver": metrics.sbs_solver,
            "sbs_cost": metrics.sbs_cost,
            "vbs_cost": metrics.vbs_cost,
            "policy_cost": metrics.policy_cost,
            "gap_closure": metrics.gap_closure,
        },
        "policy_to_sbs_ratio_bootstrap": bootstrap_ratio(policy, sbs),
        "median_winner_counts": {
            "ds": sum(r["median_winner"] == "ds" for r in rows),
            "ipm": sum(r["median_winner"] == "ipm" for r in rows),
        },
        "instances_with_at_least_4_of_5_same_round_winner": stable_instances,
        "by_family": by_family,
        "folds": folds,
        "max_objective_abs_diff": max(float(r["objective_abs_diff"]) for r in rows),
        "environment": {"python": sys.version, "numpy": np.__version__, "scipy": scipy.__version__, "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "wall_s": perf_counter() - started,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({
        "instances": payload["instances"],
        "solver_calls": payload["solver_calls"],
        "median_winner_counts": payload["median_winner_counts"],
        "stable_instances": stable_instances,
        "gap_closure": payload["metrics"]["gap_closure"],
        "policy_to_sbs_ratio": payload["policy_to_sbs_ratio_bootstrap"],
        "wall_s": payload["wall_s"],
    }, indent=2))


if __name__ == "__main__":
    main()
