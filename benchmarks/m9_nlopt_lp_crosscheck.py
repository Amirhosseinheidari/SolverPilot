from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import nlopt
import numpy as np
import scipy

from solverpilot import LinearProblem
from solverpilot.backends import NLoptNativeBackend, ScipyHighsLPBackend
from solverpilot.runtime import execute
from solverpilot.validate import PublicStatus

OUT = Path(__file__).parent / "results" / "m9-nlopt-lp-crosscheck.json"
CASES = 200


def make_case(seed: int) -> LinearProblem:
    rng = np.random.default_rng(88000 + seed)
    n = 6 + seed % 9
    m = 10 + seed % 11
    A = rng.normal(size=(m, n))
    x0 = rng.uniform(0.15, 0.85, size=n)
    ax = A @ x0
    slack = rng.uniform(0.2, 1.0, size=m)
    lo = ax - slack
    hi = ax + slack
    for i in range(m):
        # Exercise equality, lower-only, upper-only and ranged rows.
        if i % 7 == 0 and seed % 2 == 0:
            lo[i] = hi[i] = ax[i]
        elif i % 3 == 0:
            lo[i] = -np.inf
        elif i % 3 == 1:
            hi[i] = np.inf
    return LinearProblem.from_data(
        A=A,
        c=rng.normal(size=n),
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=lo,
        constraint_upper=hi,
        metadata={"m9_seed": seed},
    )


def main() -> None:
    highs = ScipyHighsLPBackend(method="highs-ds")
    nl = NLoptNativeBackend(max_evals=8000, ftol_rel=1e-11, xtol_rel=1e-11, constraint_tol=1e-10)
    # Warm libraries outside recorded loop.
    warm = make_case(999)
    execute(warm, highs); execute(warm, nl)

    rows = []
    failures = 0
    max_obj_diff = 0.0
    max_nlopt_violation = 0.0
    started = perf_counter()
    for seed in range(CASES):
        p = make_case(seed)
        rh = execute(p, highs)
        rn = execute(p, nl)
        valid_h = rh.status is PublicStatus.VALID_OPTIMAL and bool(rh.validation and rh.validation.valid)
        valid_n = bool(rn.validation and rn.validation.valid)
        obj_diff = None
        if rh.objective is not None and rn.objective is not None:
            obj_diff = abs(float(rh.objective) - float(rn.objective))
            max_obj_diff = max(max_obj_diff, obj_diff)
        violation = None if rn.validation is None else max(
            float(rn.validation.max_bound_violation),
            float(rn.validation.max_constraint_violation),
        )
        if violation is not None:
            max_nlopt_violation = max(max_nlopt_violation, violation)
        match = bool(
            valid_h and valid_n and obj_diff is not None
            and obj_diff <= 1e-7 * max(1.0, abs(float(rh.objective)))
        )
        if not match:
            failures += 1
        rows.append({
            "seed": seed,
            "n": p.n_variables,
            "m": p.n_constraints,
            "highs_status": rh.status.value,
            "nlopt_status": rn.status.value,
            "highs_objective": rh.objective,
            "nlopt_objective": rn.objective,
            "objective_abs_diff": obj_diff,
            "nlopt_validation_valid": valid_n,
            "nlopt_max_violation": violation,
            "match": match,
        })

    payload = {
        "benchmark": "M9 cross-library continuous-LP correctness: HiGHS DS vs NLopt SLSQP",
        "cases": CASES,
        "passed": CASES - failures,
        "failed": failures,
        "max_objective_abs_diff": max_obj_diff,
        "max_nlopt_feasibility_violation": max_nlopt_violation,
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
        "cases", "passed", "failed", "max_objective_abs_diff",
        "max_nlopt_feasibility_violation", "wall_s")}, indent=2))


if __name__ == "__main__":
    main()
