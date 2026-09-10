from __future__ import annotations

import json
from pathlib import Path
import platform
import sys
import tempfile
from time import perf_counter

import numpy as np
import scipy

from solverpilot import CandidateSolution, PublicStatus, execute, read_mps, validate_solution
from solverpilot.backends import ScipyVendoredHighsDevBackend

OUT = Path(__file__).parent / "results" / "m5-mps-parser-crosscheck.json"
CASES = 200


def _fmt(v: float) -> str:
    # Mix E and D exponents to exercise legacy numeric parsing.
    s = f"{float(v):.12E}"
    return s.replace("E", "D") if int(abs(v) * 1000) % 3 == 0 else s


def build_case(seed: int) -> str:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(4, 13))
    m = int(rng.integers(2, 9))
    n_int = int(rng.integers(0, max(1, n // 2 + 1)))
    n_bin = int(rng.integers(0, n_int + 1))

    # Finite bounds make every generated instance bounded. Integer variables come first
    # so one marker block can represent them without changing semantics.
    lowers = np.zeros(n)
    uppers = rng.integers(2, 9, size=n).astype(float)
    if n_bin:
        uppers[:n_bin] = 1.0
    x_feas = rng.uniform(lowers, uppers)
    if n_int:
        x_feas[:n_int] = np.floor(x_feas[:n_int] + 1e-12)

    A = rng.normal(size=(m, n))
    A[rng.random(size=A.shape) < 0.45] = 0.0
    # Ensure no empty rows because the cross-check is about format semantics, not presolve.
    for i in range(m):
        if not np.any(A[i]):
            A[i, int(rng.integers(0, n))] = float(rng.choice([-1.0, 1.0]))
    lhs = A @ x_feas
    row_types = rng.choice(np.array(["L", "G", "E"]), size=m, p=[0.4, 0.4, 0.2])
    rhs = np.zeros(m)
    ranges: dict[int, float] = {}
    for i, t in enumerate(row_types):
        use_range = bool(rng.random() < 0.30)
        width = float(rng.uniform(0.5, 3.0))
        slack = float(rng.uniform(0.2, 3.0))
        if use_range:
            if t == "G":
                rhs[i] = lhs[i] - width / 2
                ranges[i] = width * (-1 if rng.random() < 0.5 else 1)
            elif t == "L":
                rhs[i] = lhs[i] + width / 2
                ranges[i] = width * (-1 if rng.random() < 0.5 else 1)
            else:
                if rng.random() < 0.5:
                    rhs[i] = lhs[i] - width / 2
                    ranges[i] = width
                else:
                    rhs[i] = lhs[i] + width / 2
                    ranges[i] = -width
        else:
            if t == "G":
                rhs[i] = lhs[i] - slack
            elif t == "L":
                rhs[i] = lhs[i] + slack
            else:
                rhs[i] = lhs[i]

    c = rng.normal(size=n)
    sense = "MAX" if rng.random() < 0.5 else "MIN"
    offset = float(rng.uniform(-5, 5))

    lines = [f"NAME CASE{seed}", "OBJSENSE", f" {sense}", "ROWS", " N OBJ"]
    for i, t in enumerate(row_types):
        lines.append(f" {t} R{i}")
    lines.append("COLUMNS")
    if n_int:
        lines.append("    MARK0 'MARKER' 'INTORG'")
    for j in range(n):
        if j == n_int and n_int:
            lines.append("    MARK1 'MARKER' 'INTEND'")
        pairs = [("OBJ", c[j])] + [(f"R{i}", A[i, j]) for i in range(m) if A[i, j] != 0.0]
        for row, value in pairs:
            lines.append(f"    X{j} {row} {_fmt(value)}")
    if n_int == n and n_int:
        lines.append("    MARK1 'MARKER' 'INTEND'")

    lines.append("RHS")
    for i, value in enumerate(rhs):
        lines.append(f"    RHS1 R{i} {_fmt(value)}")
    lines.append(f"    RHS1 OBJ {_fmt(-offset)}")

    if ranges:
        lines.append("RANGES")
        for i, value in ranges.items():
            lines.append(f"    RNG1 R{i} {_fmt(value)}")

    lines.append("BOUNDS")
    for j in range(n):
        if j < n_bin:
            lines.append(f" BV BND X{j}")
        elif j < n_int:
            lines.append(f" LI BND X{j} 0")
            lines.append(f" UI BND X{j} {int(uppers[j])}")
        else:
            lines.append(f" UP BND X{j} {_fmt(uppers[j])}")
    lines.append("ENDATA")
    return "\n".join(lines) + "\n"


def main() -> None:
    from scipy.optimize._highspy._core import _Highs

    started = perf_counter()
    failures: list[dict[str, object]] = []
    rows: list[dict[str, object]] = []
    with tempfile.TemporaryDirectory(prefix="optimind-m5-mps-") as tmp:
        root = Path(tmp)
        for case in range(CASES):
            seed = 910000 + case
            text = build_case(seed)
            path = root / f"case-{case}.mps"
            path.write_text(text, encoding="ascii")
            problem = read_mps(path)

            ours = execute(problem, ScipyVendoredHighsDevBackend())
            h = _Highs()
            h.setOptionValue("output_flag", False)
            read_status = h.readModel(str(path))
            run_status = h.run()
            direct_model_status = h.modelStatusToString(h.getModelStatus())
            direct_obj = float(h.getObjectiveValue())
            direct_x = np.asarray(h.getSolution().col_value, dtype=np.float64)
            direct_validation = validate_solution(
                problem,
                CandidateSolution(x=direct_x, objective_reported=direct_obj),
            )

            objective_diff = None
            if ours.objective is not None:
                objective_diff = abs(float(ours.objective) - direct_obj)
            ok = (
                ours.status is PublicStatus.VALID_OPTIMAL
                and direct_validation.valid
                and objective_diff is not None
                and objective_diff <= 1e-7 * max(1.0, abs(direct_obj))
            )
            row = {
                "case": case,
                "seed": seed,
                "problem_class": ours.trace.problem_class,
                "n": problem.n_variables,
                "m": problem.n_constraints,
                "nnz": problem.nnz,
                "read_status": str(read_status),
                "run_status": str(run_status),
                "direct_model_status": str(direct_model_status),
                "ours_status": ours.status.value,
                "ours_validation": None if ours.validation is None else ours.validation.valid,
                "direct_validation": direct_validation.valid,
                "objective_diff": objective_diff,
                "ok": ok,
            }
            rows.append(row)
            if not ok:
                failures.append(row | {"mps": text})
                # Keep investigating all cases; do not stop at the first discrepancy.

    payload = {
        "benchmark": "M5 randomized MPS semantic cross-check against SciPy-vendored native HiGHS file reader",
        "cases": CASES,
        "passed": CASES - len(failures),
        "failed": len(failures),
        "max_objective_abs_diff": max((float(r["objective_diff"]) for r in rows if r["objective_diff"] is not None), default=None),
        "environment": {
            "python": sys.version,
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "platform": platform.platform(),
        },
        "wall_s": perf_counter() - started,
        "failures": failures,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("cases", "passed", "failed", "max_objective_abs_diff", "wall_s")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
