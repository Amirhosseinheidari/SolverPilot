from __future__ import annotations

import json
import math
import os
import platform
import sys
from dataclasses import asdict
from pathlib import Path
from statistics import mean, median
from time import perf_counter

import numpy as np
import scipy
from scipy import sparse

from solverpilot import LinearProblem, execute
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.evaluation import performance_profile, portfolio_metrics
from solverpilot.inspect import inspect_problem
from solverpilot.validate import PublicStatus

OUT = Path(__file__).resolve().parent / "results" / "m4-lp-selector-holdout.json"

# Families are declared before any timing results are observed. They are designed to
# vary aspect ratio, density, and equality structure without encoding solver outcomes.
FAMILIES = {
    "tall_moderate": dict(n=300, m=800, density=0.025, equality_fraction=0.05),
    "tall_strong": dict(n=220, m=900, density=0.025, equality_fraction=0.05),
    "wide_moderate": dict(n=800, m=300, density=0.025, equality_fraction=0.05),
    "balanced_sparse": dict(n=500, m=500, density=0.015, equality_fraction=0.05),
    "balanced_dense": dict(n=250, m=250, density=0.25, equality_fraction=0.05),
    "equality_heavy": dict(n=400, m=400, density=0.03, equality_fraction=0.50),
}
SEEDS_PER_FAMILY = 4
REPEATS = 3
METHODS = ("scipy-highs-ds", "scipy-highs-ipm")


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
    lower = ax0 - slack
    upper = ax0 + slack
    eq_count = int(round(cfg["equality_fraction"] * m))
    if eq_count:
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


def solve_repeated(problem: LinearProblem, method: str, repeats: int) -> dict[str, object]:
    backend = ScipyHighsLPBackend(method=method)
    times: list[float] = []
    objectives: list[float] = []
    iterations: list[int] = []
    for _ in range(repeats):
        r = execute(problem, backend)
        if r.status is not PublicStatus.VALID_OPTIMAL:
            raise AssertionError(f"{method} failed with {r.status}")
        times.append(float(r.trace.timings.total_s))
        objectives.append(float(r.objective))
        iterations.append(int((r.raw_statistics or {}).get("nit", 0)))
    return {
        "cost_s": float(median(times)),
        "samples_s": times,
        "objective": float(median(objectives)),
        "iterations_median": int(median(iterations)),
    }


def rule_candidates() -> dict[str, float | None]:
    # None means always DS. Threshold t means choose IPM iff m/n >= t.
    # Always-IPM is encoded as 0.0.
    return {
        "always_ds": None,
        "always_ipm": 0.0,
        "ipm_if_m_over_n_ge_1.50": 1.50,
        "ipm_if_m_over_n_ge_2.00": 2.00,
        "ipm_if_m_over_n_ge_2.50": 2.50,
        "ipm_if_m_over_n_ge_3.00": 3.00,
    }


def choose_for_rule(rule_threshold: float | None, row: dict[str, object]) -> str:
    if rule_threshold is None:
        return "scipy-highs-ds"
    if rule_threshold == 0.0:
        return "scipy-highs-ipm"
    ratio = float(row["m"]) / float(row["n"])
    return "scipy-highs-ipm" if ratio >= rule_threshold else "scipy-highs-ds"


def training_cost(rule_threshold: float | None, rows: list[dict[str, object]]) -> float:
    values = []
    for row in rows:
        choice = choose_for_rule(rule_threshold, row)
        key = "ds" if choice.endswith("ds") else "ipm"
        values.append(float(row[key]["cost_s"]))
    return float(mean(values))


def bootstrap_mean_ratio(policy: np.ndarray, baseline: np.ndarray, seed: int = 20260831, draws: int = 5000) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    n = policy.size
    ratios = np.empty(draws, dtype=np.float64)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        denom = float(np.mean(baseline[idx]))
        ratios[i] = float(np.mean(policy[idx]) / denom) if denom > 0 else math.inf
    lo, hi = np.quantile(ratios, [0.025, 0.975])
    return {
        "point": float(np.mean(policy) / np.mean(baseline)),
        "bootstrap_95_low": float(lo),
        "bootstrap_95_high": float(hi),
        "draws": draws,
    }


def main() -> None:
    # Warm each method outside measurements.
    warm = make_feasible_lp(990001, "balanced_sparse")
    execute(warm, ScipyHighsLPBackend("highs-ds"))
    execute(warm, ScipyHighsLPBackend("highs-ipm"))

    rows: list[dict[str, object]] = []
    started = perf_counter()
    for family_index, family in enumerate(FAMILIES):
        for local_seed in range(SEEDS_PER_FAMILY):
            seed = 73000 + family_index * 100 + local_seed
            p = make_feasible_lp(seed, family)

            # Measure cheap inspection separately because a deployable selector must pay it.
            inspect_samples = []
            fingerprint = None
            for _ in range(REPEATS):
                t0 = perf_counter()
                fingerprint = inspect_problem(p)
                inspect_samples.append(perf_counter() - t0)
            assert fingerprint is not None

            order = ["highs-ds", "highs-ipm"]
            if seed % 2:
                order.reverse()
            direct: dict[str, dict[str, object]] = {}
            for method in order:
                direct["ds" if method.endswith("ds") else "ipm"] = solve_repeated(p, method, REPEATS)

            objective_diff = abs(float(direct["ds"]["objective"]) - float(direct["ipm"]["objective"]))
            if objective_diff > 1e-6 * max(1.0, abs(float(direct["ds"]["objective"]))):
                raise AssertionError(f"objective disagreement on {family}/{seed}: {objective_diff}")

            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "n": p.n_variables,
                    "m": p.n_constraints,
                    "nnz": int(p.A.nnz),
                    "density": float(p.A.nnz / (p.n_variables * p.n_constraints)),
                    "m_over_n": float(p.n_constraints / p.n_variables),
                    "inspect_s": float(median(inspect_samples)),
                    "objective_abs_diff": objective_diff,
                    "ds": direct["ds"],
                    "ipm": direct["ipm"],
                }
            )

    # Leave-one-family-out: rule tuning never sees the held-out structural family.
    candidates = rule_candidates()
    fold_records = []
    policy_choice_by_key: dict[tuple[str, int], str] = {}
    for held_out in FAMILIES:
        train = [r for r in rows if r["family"] != held_out]
        test = [r for r in rows if r["family"] == held_out]
        scored = [(training_cost(threshold, train), name, threshold) for name, threshold in candidates.items()]
        train_cost, rule_name, threshold = min(scored, key=lambda x: (x[0], x[1]))
        choices = []
        for row in test:
            choice = choose_for_rule(threshold, row)
            policy_choice_by_key[(str(row["family"]), int(row["seed"]))] = choice
            choices.append(choice)
        fold_records.append(
            {
                "held_out_family": held_out,
                "selected_rule": rule_name,
                "threshold": threshold,
                "training_mean_cost_s": train_cost,
                "test_choices": choices,
            }
        )

    costs = {
        "scipy-highs-ds": [float(r["ds"]["cost_s"]) for r in rows],
        "scipy-highs-ipm": [float(r["ipm"]["cost_s"]) for r in rows],
    }
    policy_choices = [policy_choice_by_key[(str(r["family"]), int(r["seed"]))] for r in rows]
    policy_costs = []
    for row, choice in zip(rows, policy_choices):
        key = "ds" if choice.endswith("ds") else "ipm"
        policy_costs.append(float(row[key]["cost_s"]) + float(row["inspect_s"]))

    metrics = portfolio_metrics(costs, policy_choices, policy_costs=policy_costs)
    taus = [1.0, 1.05, 1.10, 1.25, 1.5, 2.0, 3.0]
    profile = performance_profile(costs, taus)

    sbs_values = np.asarray(costs[metrics.sbs_solver], dtype=np.float64)
    policy_values = np.asarray(policy_costs, dtype=np.float64)
    ratio_ci = bootstrap_mean_ratio(policy_values, sbs_values)

    winner_counts = {
        "scipy-highs-ds": sum(float(r["ds"]["cost_s"]) < float(r["ipm"]["cost_s"]) for r in rows),
        "scipy-highs-ipm": sum(float(r["ipm"]["cost_s"]) < float(r["ds"]["cost_s"]) for r in rows),
    }
    by_family = {}
    for family in FAMILIES:
        fam = [r for r in rows if r["family"] == family]
        by_family[family] = {
            "instances": len(fam),
            "ds_median_s": float(median(float(r["ds"]["cost_s"]) for r in fam)),
            "ipm_median_s": float(median(float(r["ipm"]["cost_s"]) for r in fam)),
            "ds_wins": sum(float(r["ds"]["cost_s"]) < float(r["ipm"]["cost_s"]) for r in fam),
            "ipm_wins": sum(float(r["ipm"]["cost_s"]) < float(r["ds"]["cost_s"]) for r in fam),
            "policy_ipm_choices": sum(policy_choice_by_key[(family, int(r["seed"]))] == "scipy-highs-ipm" for r in fam),
        }

    payload = {
        "benchmark": "M4 controlled LP structural holdout; synthetic research evidence only",
        "selection_protocol": "leave-one-family-out; candidate rules fixed in source; policy pays inspection time",
        "families": FAMILIES,
        "instances": len(rows),
        "repeats_per_solver_instance": REPEATS,
        "metrics": asdict(metrics),
        "policy_to_sbs_mean_cost_ratio": ratio_ci,
        "performance_profile": {"taus": taus, "fractions": profile},
        "winner_counts": winner_counts,
        "by_family": by_family,
        "folds": fold_records,
        "max_objective_abs_diff": max(float(r["objective_abs_diff"]) for r in rows),
        "policy_choice_counts": {m: policy_choices.count(m) for m in METHODS},
        "environment": {
            "python": sys.version,
            "scipy": scipy.__version__,
            "numpy": np.__version__,
            "platform": platform.platform(),
            "machine": platform.machine(),
            "cpu_count": os.cpu_count(),
        },
        "rows": rows,
        "wall_s": perf_counter() - started,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    summary_keys = [
        "instances",
        "metrics",
        "policy_to_sbs_mean_cost_ratio",
        "winner_counts",
        "by_family",
        "folds",
        "policy_choice_counts",
        "max_objective_abs_diff",
        "wall_s",
    ]
    print(json.dumps({k: payload[k] for k in summary_keys}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
