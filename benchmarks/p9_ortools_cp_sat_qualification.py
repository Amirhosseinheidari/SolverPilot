from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import random
import sys

import solverpilot as om

TARGET_VERSION = "9.15.6755"
TARGET_WHEEL = "ortools-9.15.6755-cp313-cp313-manylinux_2_27_x86_64.manylinux_2_28_x86_64.whl"
TARGET_WHEEL_SHA256 = "ebd5aea00374e3aad7a78de59058aca5e871a26a3c385cd0860ef1d685d03c9a"
TARGET_WHEEL_URL = (
    "https://files.pythonhosted.org/packages/08/b9/"
    "28d5efb832190b6edfccc5a703e88e64779c1eda34a42ea96d03307236c0/"
    + TARGET_WHEEL
)


def _result_key(r):
    if r.status == "infeasible":
        return ("infeasible", None)
    if r.assignment is None:
        return (r.status, None)
    return ("feasible", r.objective)


def _compare(model: om.CPModel, name: str):
    p = model.compile()
    ref = p and model.solve(om.ReferenceCPBackend(max_states=3_000_000))
    sat = model.solve(om.ORToolsCPSATBackend(num_workers=1))
    same_status = (ref.status == "infeasible" and sat.status == "infeasible") or (
        ref.assignment is not None and sat.assignment is not None
    )
    same_obj = ref.objective == sat.objective
    sat_valid = sat.assignment is None or sat.validation.valid
    proof_ok = sat.optimality_proven if model.compile().objective is not None else (
        sat.status in {"optimal", "feasible"} or sat.status == "infeasible"
    )
    passed = bool(same_status and same_obj and sat_valid and proof_ok)
    return {
        "name": name,
        "passed": passed,
        "reference": {"status": ref.status, "objective": ref.objective},
        "cp_sat": {
            "status": sat.status,
            "objective": sat.objective,
            "optimality_proven": sat.optimality_proven,
            "validation_valid": sat.validation.valid,
            "raw": sat.raw_statistics,
        },
    }


def deterministic_cases():
    out = []

    m = om.CPModel("linear-min")
    x = m.int_var(0, 4, "x"); y = m.int_var(0, 4, "y")
    m.add(x + y >= 5); m.minimize(2 * x + y)
    out.append(("linear-objective", m))

    m = om.CPModel("all-different")
    xs = [m.int_var(1, 3, f"x{i}") for i in range(3)]
    m.add_all_different(xs); m.add(xs[0] == 1); m.minimize(xs[2])
    out.append(("all-different", m))

    m = om.CPModel("exactly-one")
    bs = [m.bool_var(f"b{i}") for i in range(4)]
    m.add_exactly_one(bs); m.maximize(sum((i + 1) * b for i, b in enumerate(bs)))
    out.append(("exactly-one", m))

    m = om.CPModel("table")
    x = m.int_var(0, 2, "x"); y = m.int_var(0, 2, "y")
    m.add_allowed_assignments([x, y], [(0, 2), (2, 0)]); m.minimize(x + y)
    out.append(("allowed-table", m))

    m = om.CPModel("element")
    i = m.int_var(0, 2, "i"); t = m.int_var(0, 10, "t")
    m.add_element(i, [4, 7, 9], t); m.add(i == 1); m.minimize(t)
    out.append(("element", m))

    m = om.CPModel("circuit")
    lits = {}
    arcs = []
    for a, b in [(0, 1), (1, 2), (2, 0), (0, 0), (1, 1), (2, 2)]:
        lits[a, b] = m.bool_var(f"a{a}{b}"); arcs.append((a, b, lits[a, b]))
    m.add_circuit(arcs)
    for e in [(0, 1), (1, 2), (2, 0)]: m.add(lits[e] == 1)
    for e in [(0, 0), (1, 1), (2, 2)]: m.add(lits[e] == 0)
    out.append(("circuit", m))

    m = om.CPModel("circuit-self-loops")
    loops = []
    for i in range(3):
        b = m.bool_var(f"l{i}"); loops.append((i, i, b)); m.add(b == 1)
    m.add_circuit(loops)
    out.append(("circuit-empty-tour-self-loops", m))

    m = om.CPModel("no-overlap")
    s1 = m.int_var(0, 2, "s1"); s2 = m.int_var(0, 2, "s2")
    a = m.interval_var(s1, 2, "a"); b = m.interval_var(s2, 2, "b")
    m.add_no_overlap([a, b]); m.minimize(s1 + s2)
    out.append(("no-overlap", m))

    m = om.CPModel("cumulative")
    starts = [m.int_var(0, 2, f"s{i}") for i in range(3)]
    iv = [m.interval_var(s, 2, f"i{i}") for i, s in enumerate(starts)]
    m.add_cumulative(iv, [2, 2, 1], 3); m.minimize(sum(starts))
    out.append(("cumulative", m))

    m = om.CPModel("infeasible")
    x = m.int_var(0, 1, "x"); m.add(x >= 2)
    out.append(("infeasible-proof", m))

    return out


def random_linear_cases(rng: random.Random, n_cases: int):
    for k in range(n_cases):
        m = om.CPModel(f"random-linear-{k}")
        xs = [m.int_var(0, 3, f"x{i}") for i in range(3)]
        coeff = [rng.randint(-2, 3) or 1 for _ in xs]
        rhs = rng.randint(0, 5)
        m.add(sum(c * x for c, x in zip(coeff, xs)) >= rhs)
        if k % 3 == 0:
            m.add_all_different(xs)
        weights = [rng.randint(-3, 4) for _ in xs]
        m.minimize(sum(w * x for w, x in zip(weights, xs)))
        yield f"random-linear-{k}", m


def random_exactly_one_cases(rng: random.Random, n_cases: int):
    for k in range(n_cases):
        m = om.CPModel(f"random-exactly-one-{k}")
        bs = [m.bool_var(f"b{i}") for i in range(4)]
        m.add_exactly_one(bs)
        weights = [rng.randint(-5, 8) for _ in bs]
        m.minimize(sum(w * b for w, b in zip(weights, bs)))
        yield f"random-exactly-one-{k}", m


def random_table_element_cases(rng: random.Random, n_cases: int):
    for k in range(n_cases):
        m = om.CPModel(f"random-table-element-{k}")
        i = m.int_var(0, 2, "i")
        vals = [rng.randint(0, 9) for _ in range(3)]
        t = m.int_var(min(vals), max(vals), "t")
        m.add_element(i, vals, t)
        allowed = [(j, vals[j]) for j in range(3) if rng.random() < 0.8]
        if not allowed: allowed = [(0, vals[0])]
        m.add_allowed_assignments([i, t], allowed)
        m.minimize(t)
        yield f"random-table-element-{k}", m


def random_scheduling_cases(rng: random.Random, n_cases: int):
    for k in range(n_cases):
        m = om.CPModel(f"random-schedule-{k}")
        n = 2 + (k % 2)
        starts = [m.int_var(0, 4, f"s{i}") for i in range(n)]
        durations = [rng.randint(1, 2) for _ in range(n)]
        intervals = [m.interval_var(s, d, f"i{i}") for i, (s, d) in enumerate(zip(starts, durations))]
        if k % 2 == 0:
            m.add_no_overlap(intervals)
        else:
            demands = [rng.randint(1, 2) for _ in intervals]
            cap = max(demands) + 1
            m.add_cumulative(intervals, demands, cap)
        m.minimize(sum(starts))
        yield f"random-schedule-{k}", m


def random_circuit_cases(rng: random.Random, n_cases: int):
    # Complete 3-node digraph + self loops. 9 Boolean states => reference enumeration remains small.
    for k in range(n_cases):
        m = om.CPModel(f"random-circuit-{k}")
        arcs = []
        terms = []
        for i in range(3):
            for j in range(3):
                b = m.bool_var(f"a{i}_{j}")
                arcs.append((i, j, b))
                terms.append(rng.randint(-3, 5) * b)
        m.add_circuit(arcs); m.minimize(sum(terms))
        yield f"random-circuit-{k}", m


def run(seed=9156755):
    backend = om.ORToolsCPSATBackend(num_workers=1)
    binding = backend.binding_version
    base = {
        "schema": "solverpilot.p9.ortools-cp-sat-qualification.v1",
        "target_version": TARGET_VERSION,
        "target_wheel": TARGET_WHEEL,
        "target_wheel_sha256": TARGET_WHEEL_SHA256,
        "target_wheel_url": TARGET_WHEEL_URL,
        "binding_version": binding,
        "available_exact_version": backend.is_available(),
        "seed": seed,
    }
    if not backend.is_available():
        base.update({
            "classification": "not_executed_fail_closed",
            "passed": False,
            "executed_cases": 0,
            "reason": "OR-Tools exact verified version is unavailable in this runtime",
        })
        return base

    import ortools
    from ortools.sat.python import cp_model
    checks = []
    for name, model in deterministic_cases():
        checks.append(_compare(model, name))
    rng = random.Random(seed)
    generators = [
        random_linear_cases(rng, 30),
        random_exactly_one_cases(rng, 20),
        random_table_element_cases(rng, 20),
        random_scheduling_cases(rng, 20),
        random_circuit_cases(rng, 20),
    ]
    for gen in generators:
        for name, model in gen:
            checks.append(_compare(model, name))
    passed = sum(1 for c in checks if c["passed"])
    base.update({
        "classification": "executed_exact_version_conformance",
        "ortools_version": ortools.__version__,
        "cp_model_module": cp_model.__file__,
        "executed_cases": len(checks),
        "passed_cases": passed,
        "failed_cases": len(checks) - passed,
        "passed": passed == len(checks),
        "checks": checks,
    })
    return base


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", default="benchmarks/results/p9-ortools-cp-sat-qualification.json")
    ap.add_argument("--require-execution", action="store_true")
    ns = ap.parse_args(argv)
    out = run()
    path = Path(ns.output); path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print(json.dumps({k: out.get(k) for k in ["classification","target_version","binding_version","executed_cases","passed_cases","failed_cases","passed","reason"] if k in out}, indent=2, sort_keys=True))
    if ns.require_execution and out["classification"] != "executed_exact_version_conformance":
        return 2
    if out["classification"] == "executed_exact_version_conformance" and not out["passed"]:
        return 3
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
