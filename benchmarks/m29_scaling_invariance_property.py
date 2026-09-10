from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np

from solverpilot import LinearProblem

SEED = 20260906
CASES = 200


def _load_worker(root: Path):
    p = root / "benchmarks/m29_feature_worker.py"
    spec = importlib.util.spec_from_file_location("m29_feature_worker_property", p)
    if spec is None or spec.loader is None:
        raise RuntimeError("cannot load M29 feature worker")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    root = Path(__file__).resolve().parents[1]
    mod = _load_worker(root)
    rng = np.random.default_rng(SEED)
    failures = []
    max_abs = 0.0
    max_rel = 0.0

    for case in range(CASES):
        m = int(rng.integers(1, 18)); n = int(rng.integers(1, 18))
        A = rng.normal(size=(m, n))
        A *= rng.random((m, n)) < rng.uniform(0.1, 0.7)
        for i in range(m):
            if not np.any(A[i] != 0):
                A[i, int(rng.integers(0, n))] = float(rng.choice([-1, 1])) * 10 ** rng.uniform(-2, 2)
        c = rng.normal(size=n)
        if not np.any(c != 0): c[0] = 1.0
        vl = np.full(n, -np.inf); vu = np.full(n, np.inf)
        for j in range(n):
            t = int(rng.integers(0, 4))
            if t == 1: vl[j] = rng.uniform(-10, 0)
            elif t == 2: vu[j] = rng.uniform(0, 10)
            elif t == 3:
                vl[j] = rng.uniform(-10, 0); vu[j] = rng.uniform(0, 10)
        cl = np.full(m, -np.inf); cu = np.full(m, np.inf)
        for i in range(m):
            t = int(rng.integers(1, 4))
            if t == 1: cu[i] = rng.uniform(-5, 15)
            elif t == 2: cl[i] = rng.uniform(-15, 5)
            else:
                v = rng.uniform(-5, 5); cl[i] = v; cu[i] = v
        p = LinearProblem.from_data(A=A, c=c, variable_lower=vl, variable_upper=vu, constraint_lower=cl, constraint_upper=cu)

        row_scale = 10 ** rng.uniform(-8, 8, size=m)
        obj_scale = float(10 ** rng.uniform(-8, 8))
        cl2 = cl.copy(); cu2 = cu.copy()
        fl = np.isfinite(cl2); fu = np.isfinite(cu2)
        cl2[fl] *= row_scale[fl]; cu2[fu] *= row_scale[fu]
        q = LinearProblem.from_data(
            A=A * row_scale[:, None], c=c * obj_scale,
            variable_lower=vl, variable_upper=vu,
            constraint_lower=cl2, constraint_upper=cu2,
        )
        fa = mod.static_features(p); fb = mod.static_features(q)
        for key in fa:
            av = float(fa[key]); bv = float(fb[key])
            diff = abs(av - bv); rel = diff / max(1.0, abs(av), abs(bv))
            max_abs = max(max_abs, diff); max_rel = max(max_rel, rel)
            if not np.isclose(av, bv, rtol=1e-8, atol=1e-9):
                failures.append({"case": case, "feature": key, "a": av, "b": bv, "abs_diff": diff, "relative_diff": rel})
                break

    payload = {
        "schema": "optimind.m29.scaling_invariance_property.v1",
        "seed": SEED,
        "cases": CASES,
        "passed": CASES - len(failures),
        "failed": len(failures),
        "max_abs_difference": max_abs,
        "max_relative_difference": max_rel,
        "contract": "M29 canonical static features are invariant to positive independent row scaling plus positive objective scaling",
        "failures": failures[:20],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in ("cases", "passed", "failed", "max_abs_difference", "max_relative_difference")}, indent=2))
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
