from __future__ import annotations

import json
import math
from dataclasses import asdict
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse
from sklearn.tree import DecisionTreeClassifier, export_text

from solverpilot import LinearProblem, execute, inspect_problem, solve
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.evaluation import performance_profile, portfolio_metrics
from solverpilot.validate import PublicStatus

OUT = Path(__file__).resolve().parent / "results" / "m2-selector-research.json"

# These are deliberately generator families rather than duplicated siblings randomly
# split across train/test. The benchmark is still synthetic and only research evidence.
TRAIN_FAMILIES = {
    "small_sparse": (220, 220, 0.020, 0.05),
    "large_sparse_balanced": (1000, 1000, 0.006, 0.10),
    "large_sparse_tall": (400, 1300, 0.012, 0.10),
    "dense_mid": (400, 400, 0.200, 0.25),
}
TEST_FAMILIES = {
    "holdout_sparse_balanced": (900, 900, 0.007, 0.10),
    "holdout_sparse_wide": (1300, 400, 0.012, 0.08),
    "holdout_dense": (450, 450, 0.180, 0.20),
    "holdout_small_wide": (500, 100, 0.015, 0.02),
}


def make_problem(seed: int, family: str, spec) -> LinearProblem:
    n, m, density, equality_fraction = spec
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=density,
        random_state=rng,
        data_rvs=lambda k: rng.normal(size=k),
        format="csr",
    )
    A = A.tolil()
    for i in range(m):
        if not A.rows[i]:
            j = int(rng.integers(0, n))
            A[i, j] = float(rng.normal() or 1.0)
    A = A.tocsr()
    x0 = rng.uniform(0.15, 0.85, size=n)
    ax0 = np.asarray(A @ x0).ravel()
    slack = rng.uniform(0.2, 0.8, size=m)
    lo = ax0 - slack
    hi = ax0 + slack
    eq_count = int(round(m * equality_fraction))
    if eq_count:
        lo[:eq_count] = ax0[:eq_count]
        hi[:eq_count] = ax0[:eq_count]
    return LinearProblem.from_data(
        A=A,
        c=rng.normal(size=n),
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lo,
        constraint_upper=hi,
        metadata={"family": family, "seed": seed},
    )


def feature_vector(problem: LinearProblem):
    t0 = perf_counter()
    f = inspect_problem(problem)
    elapsed = perf_counter() - t0
    values = np.array(
        [
            math.log10(max(f.n_variables, 1)),
            math.log10(max(f.n_constraints, 1)),
            math.log10(max(f.nnz_a, 1)),
            math.log10(max(f.density_a, 1e-12)),
            f.equality_fraction,
            f.row_nnz.mean,
            f.col_nnz.mean,
            math.log10(max(f.estimated_numeric_bytes, 1)),
        ],
        dtype=float,
    )
    return values, elapsed


FEATURE_NAMES = [
    "log10_n",
    "log10_m",
    "log10_nnz",
    "log10_density",
    "equality_fraction",
    "row_nnz_mean",
    "col_nnz_mean",
    "log10_estimated_bytes",
]


def direct_cost(problem: LinearProblem, method: str, repeats: int = 1):
    times = []
    objectives = []
    for _ in range(repeats):
        r = execute(problem, ScipyHighsLPBackend(method))
        if r.status is not PublicStatus.VALID_OPTIMAL:
            raise AssertionError((method, r.status))
        times.append(float(r.trace.timings.total_s))
        objectives.append(float(r.objective))
    return median(times), median(objectives)


def collect(families, *, base_seed: int, per_family: int):
    rows = []
    for fi, (family, spec) in enumerate(families.items()):
        for j in range(per_family):
            seed = base_seed + 100 * fi + j
            p = make_problem(seed, family, spec)
            feat, inspect_s = feature_vector(p)
            methods = ["highs-ds", "highs-ipm"]
            if seed % 2:
                methods.reverse()
            costs = {}
            objs = {}
            for method in methods:
                costs[method], objs[method] = direct_cost(p, method)
            if abs(objs["highs-ds"] - objs["highs-ipm"]) > 1e-6 * max(1.0, abs(objs["highs-ds"]), abs(objs["highs-ipm"])):
                raise AssertionError("solver objective disagreement")
            rows.append(
                {
                    "family": family,
                    "seed": seed,
                    "features": feat.tolist(),
                    "inspect_s": inspect_s,
                    "ds_s": costs["highs-ds"],
                    "ipm_s": costs["highs-ipm"],
                    "winner": "scipy-highs-ds" if costs["highs-ds"] <= costs["highs-ipm"] else "scipy-highs-ipm",
                }
            )
    return rows


def main():
    # Warm the solver paths outside measured instances.
    warm = make_problem(770001, "warm", (300, 300, 0.02, 0.05))
    execute(warm, ScipyHighsLPBackend("highs-ds"))
    execute(warm, ScipyHighsLPBackend("highs-ipm"))

    t0 = perf_counter()
    train = collect(TRAIN_FAMILIES, base_seed=51000, per_family=2)
    test = collect(TEST_FAMILIES, base_seed=61000, per_family=2)

    X_train = np.asarray([r["features"] for r in train], dtype=float)
    y_train = np.asarray([1 if r["winner"] == "scipy-highs-ipm" else 0 for r in train])
    X_test = np.asarray([r["features"] for r in test], dtype=float)

    tree = DecisionTreeClassifier(max_depth=2, min_samples_leaf=2, random_state=0)
    tree.fit(X_train, y_train)

    tree_choices = []
    tree_policy_costs = []
    for row, features in zip(test, X_test):
        pred_t0 = perf_counter()
        pred = int(tree.predict(features.reshape(1, -1))[0])
        predict_s = perf_counter() - pred_t0
        choice = "scipy-highs-ipm" if pred else "scipy-highs-ds"
        tree_choices.append(choice)
        direct = row["ipm_s"] if pred else row["ds_s"]
        tree_policy_costs.append(direct + row["inspect_s"] + predict_s)
        row["research_tree_choice"] = choice
        row["research_tree_predict_s"] = predict_s

    # Run the actual current auto policy on held-out instances, not a reconstructed proxy.
    auto_choices = []
    auto_costs = []
    for row, (family, spec) in zip(test, [(r["family"], TEST_FAMILIES[r["family"]]) for r in test]):
        p = make_problem(row["seed"], family, spec)
        r = solve(p)
        if r.status is not PublicStatus.VALID_OPTIMAL:
            raise AssertionError(r.status)
        auto_choices.append(r.plan.selected_backend)
        auto_costs.append(float(r.trace.timings.total_s))
        row["current_auto_choice"] = r.plan.selected_backend
        row["current_auto_cost_s"] = float(r.trace.timings.total_s)

    test_costs = {
        "scipy-highs-ds": [r["ds_s"] for r in test],
        "scipy-highs-ipm": [r["ipm_s"] for r in test],
    }
    current_metrics = portfolio_metrics(test_costs, auto_choices, policy_costs=auto_costs)
    research_metrics = portfolio_metrics(test_costs, tree_choices, policy_costs=tree_policy_costs)
    taus = [1.0, 1.05, 1.1, 1.25, 1.5, 2.0, 3.0]

    payload = {
        "benchmark": "M2 grouped synthetic selector research; not production evidence",
        "feature_names": FEATURE_NAMES,
        "train_families": TRAIN_FAMILIES,
        "test_families": TEST_FAMILIES,
        "train_instances": len(train),
        "test_instances": len(test),
        "train_winners": {
            "ds": sum(r["winner"] == "scipy-highs-ds" for r in train),
            "ipm": sum(r["winner"] == "scipy-highs-ipm" for r in train),
        },
        "test_winners": {
            "ds": sum(r["winner"] == "scipy-highs-ds" for r in test),
            "ipm": sum(r["winner"] == "scipy-highs-ipm" for r in test),
        },
        "tree_rule": export_text(tree, feature_names=FEATURE_NAMES),
        "current_auto": asdict(current_metrics),
        "research_tree": asdict(research_metrics),
        "performance_profile": {
            "taus": taus,
            "fractions": performance_profile(test_costs, taus),
        },
        "train": train,
        "test": test,
        "wall_s": perf_counter() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ["train_winners", "test_winners", "tree_rule", "current_auto", "research_tree", "wall_s"]}, indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
