from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from solverpilot import LinearProblem
from solverpilot.diagnose import ViolationKind, deletion_filter_conflict, diagnose_infeasibility


def planted_row_conflict(seed: int) -> LinearProblem:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(1, 5))
    j = int(rng.integers(0, n))
    gap = float(rng.uniform(0.1, 5.0))
    center = float(rng.normal())
    lower = center + gap / 2
    upper = center - gap / 2
    rows = []
    lo = []
    hi = []
    a = np.zeros(n); a[j] = 1.0
    rows.extend([a.copy(), a.copy()]); lo.extend([lower, -np.inf]); hi.extend([np.inf, upper])
    # Add irrelevant wide rows so deletion filtering has work to do.
    for _ in range(int(rng.integers(0, 5))):
        r = rng.normal(size=n)
        rows.append(r); lo.append(-1000.0); hi.append(1000.0)
    return LinearProblem.from_data(
        A=np.asarray(rows), c=np.zeros(n),
        variable_lower=np.full(n, -np.inf), variable_upper=np.full(n, np.inf),
        constraint_lower=np.asarray(lo), constraint_upper=np.asarray(hi),
    )


def planted_integer_bound_conflict(seed: int) -> LinearProblem:
    rng = np.random.default_rng(seed)
    base = int(rng.integers(-5, 6))
    lo = base + float(rng.uniform(0.05, 0.45))
    hi = base + float(rng.uniform(0.55, 0.95))
    return LinearProblem.from_data(
        A=np.zeros((0, 1)), c=[0.0],
        variable_lower=[lo], variable_upper=[hi],
        constraint_lower=[], constraint_upper=[], domains=["integer"],
    )


def main() -> None:
    row_pass = 0
    int_pass = 0
    max_checks = 0
    for seed in range(100):
        p = planted_row_conflict(seed)
        report = diagnose_infeasibility(p, include_conflict=True)
        c = report.conflict
        assert report.confirmed_infeasible
        assert c is not None and c.complete and c.irreducible
        assert len(c.atoms) == 2
        assert {a.kind for a in c.atoms} == {ViolationKind.ROW_LOWER, ViolationKind.ROW_UPPER}
        assert report.elastic is not None and report.elastic.objective is not None and report.elastic.objective > 0
        max_checks = max(max_checks, c.checks)
        row_pass += 1
    for seed in range(100, 200):
        p = planted_integer_bound_conflict(seed)
        report = diagnose_infeasibility(p, include_conflict=True)
        c = report.conflict
        assert report.confirmed_infeasible
        assert len(report.static_issues) == 1
        assert c is not None and c.complete and c.irreducible
        assert {(a.kind, a.index) for a in c.atoms} == {
            (ViolationKind.VARIABLE_LOWER, 0),
            (ViolationKind.VARIABLE_UPPER, 0),
        }
        int_pass += 1
    out = {
        "row_conflicts": {"passed": row_pass, "total": 100},
        "integer_bound_conflicts": {"passed": int_pass, "total": 100},
        "total_passed": row_pass + int_pass,
        "total": 200,
        "max_deletion_filter_checks": max_checks,
    }
    Path("benchmarks/results/m6-diagnostics-property-checks.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
