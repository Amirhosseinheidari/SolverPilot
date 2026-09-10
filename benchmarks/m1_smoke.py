from __future__ import annotations

import json
from pathlib import Path
from statistics import median
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, QuadraticProblem, VariableDomain, execute, solve
from solverpilot.backends import ScipyHighsBackend, ScipyHighsLPBackend, ScipySLSQPQPBackend
from solverpilot.validate import PublicStatus


OUT = Path(__file__).resolve().parent / "results" / "m1-smoke.json"


def make_lp(seed: int, n: int, m: int, density: float) -> LinearProblem:
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=density,
        random_state=rng,
        data_rvs=lambda k: rng.normal(size=k),
        format="csr",
    )
    x0 = rng.uniform(0.2, 0.8, size=n)
    ax0 = np.asarray(A @ x0).ravel()
    lower = np.full(m, -np.inf)
    upper = ax0 + rng.uniform(0.25, 1.0, size=m)
    # Make a small deterministic subset equalities to exercise translation.
    eq_count = max(1, m // 20)
    lower[:eq_count] = ax0[:eq_count]
    upper[:eq_count] = ax0[:eq_count]
    c = rng.normal(size=n)
    return LinearProblem.from_data(
        A=A,
        c=c,
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lower,
        constraint_upper=upper,
    )


def timed_execute(problem, backend, repeats: int = 3):
    results = []
    for _ in range(repeats):
        result = execute(problem, backend)
        results.append(result)
    return results[-1], median(r.trace.timings.solve_s for r in results), median(
        r.trace.timings.validate_s for r in results
    )


def lp_suite():
    cases = [
        (60, 80, 0.08),
        (180, 250, 0.03),
        (400, 600, 0.012),
    ]
    rows = []
    for scale_idx, (n, m, density) in enumerate(cases):
        for rep in range(3):
            seed = 1000 + scale_idx * 100 + rep
            p = make_lp(seed, n, m, density)
            ds, ds_s, ds_val = timed_execute(p, ScipyHighsLPBackend("highs-ds"))
            ipm, ipm_s, ipm_val = timed_execute(p, ScipyHighsLPBackend("highs-ipm"))
            auto = solve(p)
            assert ds.status is PublicStatus.VALID_OPTIMAL
            assert ipm.status is PublicStatus.VALID_OPTIMAL
            assert auto.status is PublicStatus.VALID_OPTIMAL
            obj_diff = abs(float(ds.objective) - float(ipm.objective))
            rows.append(
                {
                    "seed": seed,
                    "n": n,
                    "m": m,
                    "density": density,
                    "objective_abs_diff_ds_ipm": obj_diff,
                    "ds_solve_s_median": ds_s,
                    "ipm_solve_s_median": ipm_s,
                    "ds_validate_s_median": ds_val,
                    "ipm_validate_s_median": ipm_val,
                    "auto_selected": auto.plan.selected_backend if auto.plan else None,
                    "auto_inspect_s": auto.trace.timings.inspect_s,
                    "auto_plan_s": auto.trace.timings.plan_s,
                }
            )
    return rows


def milp_suite():
    rows = []
    for seed in range(2000, 2010):
        rng = np.random.default_rng(seed)
        n = 24
        values = rng.uniform(1.0, 20.0, size=n)
        weights = rng.integers(1, 15, size=n).astype(float)
        capacity = float(np.floor(0.35 * weights.sum()))
        p = LinearProblem.from_data(
            A=[weights],
            c=-values,
            variable_lower=np.zeros(n),
            variable_upper=np.ones(n),
            constraint_lower=[-np.inf],
            constraint_upper=[capacity],
            domains=[VariableDomain.BINARY] * n,
        )
        direct, solve_s, validate_s = timed_execute(p, ScipyHighsBackend(), repeats=2)
        auto = solve(p)
        assert direct.status is PublicStatus.VALID_OPTIMAL
        assert auto.status is PublicStatus.VALID_OPTIMAL
        rows.append(
            {
                "seed": seed,
                "objective": direct.objective,
                "direct_solve_s_median": solve_s,
                "direct_validate_s_median": validate_s,
                "auto_selected": auto.plan.selected_backend if auto.plan else None,
                "auto_inspect_s": auto.trace.timings.inspect_s,
                "auto_plan_s": auto.trace.timings.plan_s,
            }
        )
    return rows


def qp_suite():
    rows = []
    backend = ScipySLSQPQPBackend(maxiter=500, ftol=1e-12)
    for seed in range(3000, 3020):
        rng = np.random.default_rng(seed)
        n = 12
        diag = rng.uniform(0.5, 4.0, size=n)
        q = rng.uniform(-1.0, 1.0, size=n)
        expected = -q / diag
        p = QuadraticProblem.from_data(
            P=sparse.diags(diag, format="csr"),
            A=np.zeros((0, n)),
            q=q,
            variable_lower=np.full(n, -5.0),
            variable_upper=np.full(n, 5.0),
            constraint_lower=[],
            constraint_upper=[],
        )
        result, solve_s, validate_s = timed_execute(p, backend, repeats=2)
        assert result.status is PublicStatus.VALID_FEASIBLE
        error = float(np.max(np.abs(result.x - expected)))
        rows.append(
            {
                "seed": seed,
                "max_x_error_vs_analytic": error,
                "solve_s_median": solve_s,
                "validate_s_median": validate_s,
            }
        )
    return rows


def summarize(lp_rows, milp_rows, qp_rows):
    lp_ds = [x["ds_solve_s_median"] for x in lp_rows]
    lp_ipm = [x["ipm_solve_s_median"] for x in lp_rows]
    planner = [x["auto_plan_s"] for x in lp_rows + milp_rows]
    inspector = [x["auto_inspect_s"] for x in lp_rows + milp_rows]
    return {
        "lp_instances": len(lp_rows),
        "milp_instances": len(milp_rows),
        "qp_instances": len(qp_rows),
        "lp_max_objective_abs_diff_ds_ipm": max(
            x["objective_abs_diff_ds_ipm"] for x in lp_rows
        ),
        "lp_ds_median_solve_s": median(lp_ds),
        "lp_ipm_median_solve_s": median(lp_ipm),
        "planner_median_s": median(planner),
        "inspector_median_s": median(inspector),
        "planner_max_s": max(planner),
        "inspector_max_s": max(inspector),
        "qp_max_x_error_vs_analytic": max(x["max_x_error_vs_analytic"] for x in qp_rows),
        "all_lp_auto_selected_ds": all(x["auto_selected"] == "scipy-highs-ds" for x in lp_rows),
        "all_milp_auto_selected_bridge": all(
            x["auto_selected"] == "scipy-highs-bridge" for x in milp_rows
        ),
    }


def main():
    t0 = perf_counter()
    lp_rows = lp_suite()
    milp_rows = milp_suite()
    qp_rows = qp_suite()
    payload = {
        "benchmark": "M1 synthetic smoke; not a market/performance claim",
        "summary": summarize(lp_rows, milp_rows, qp_rows),
        "lp": lp_rows,
        "milp": milp_rows,
        "qp": qp_rows,
        "wall_s": perf_counter() - t0,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload["summary"], indent=2))
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
