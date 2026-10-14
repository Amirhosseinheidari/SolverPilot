"""Seeded development challenge for contract checks and evidence consistency.

Expected answers come from explicit mutations and constructed feasible points,
not the production compiler or a second call to the tested validator. This is a
reproducible development challenge, not an independently held-out user study.
"""

import argparse
from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import platform

import numpy as np

from solverpilot import LinearProblem, solve
from solverpilot._identity import source_tree_sha256
from solverpilot.applications.production import (
    ProductionContract, check_production_candidate, check_production_formulation,
)
from solverpilot.runtime.evidence import evidence_bundle, verify_evidence_bundle
from solverpilot.runtime.manifest import _seal, environment
from solverpilot.benchmark.integrity import canonical_sha256


def fixture(seed, size, scale):
    rng = np.random.default_rng(seed)
    quantities = rng.integers(1, 10, size=size).astype(float)
    resources = rng.integers(1, 8, size=(max(2, size // 2), size)) * scale
    contract = ProductionContract(
        profit=rng.integers(1, 20, size=size), resources=resources,
        capacity=resources @ quantities, maximum=quantities * 2,
        objective_unit="declared_currency/declared_period",
    )
    # Independently assemble the expected LP; do not ask the template to produce
    # the validator's positive control.
    problem = LinearProblem.from_data(
        A=resources, c=contract.profit, variable_lower=contract.minimum,
        variable_upper=contract.maximum, constraint_lower=np.full(resources.shape[0], -np.inf),
        constraint_upper=contract.capacity, objective_sense="maximize",
    )
    return contract, quantities, problem


def observe(rows, family, name, expected, check, *, contract_hash=None):
    try:
        actual, error = bool(check()), None
    except Exception as exc:
        actual, error = None, f"{type(exc).__name__}: {exc}"
    rows.append({"family": family, "name": name, "expected_accept": expected,
                 "actual_accept": actual, "error": error, "contract_hash": contract_hash,
                 "passed": error is None and actual is expected})


def formulation_cases(contract, p):
    yield "independent_original", True, p
    yield "reversed_equivalent_rows", True, replace(
        p, A=-p.A, constraint_lower=-p.constraint_upper,
        constraint_upper=np.full(p.n_constraints, np.inf))
    yield "missing_capacity_row", False, replace(
        p, A=p.A[:-1], constraint_lower=p.constraint_lower[:-1], constraint_upper=p.constraint_upper[:-1])
    yield "wrong_objective_sense", False, replace(p, objective_sense="minimize")
    yield "wrong_profit", False, replace(p, c=p.c * 2)
    yield "extra_objective_constant", False, replace(p, objective_offset=1.)
    yield "capacity_relaxed", False, replace(p, constraint_upper=p.constraint_upper * 2)
    yield "minimum_dropped_below_zero", False, replace(p, variable_lower=np.full(p.n_variables, -1.))
    yield "maximum_relaxed", False, replace(p, variable_upper=p.variable_upper * 2)
    yield "unit_factor_only_in_model", False, replace(p, A=p.A * 1000)
    yield "wrong_variable_domain", False, replace(p, domains=np.full(p.n_variables, "integer"))
    yield "inequality_reversed", False, replace(
        p, constraint_lower=p.constraint_upper, constraint_upper=np.full(p.n_constraints, np.inf))


def candidate_cases(contract, x):
    objective = float(contract.profit @ x)
    yield "constructed_feasible", True, x, objective
    yield "zero_feasible", True, np.zeros_like(x), 0.
    yield "resource_overrun", False, x * 1.1, objective * 1.1
    yield "negative_quantities", False, -x, -objective
    yield "maximum_overrun", False, x * 3, objective * 3
    yield "nan_candidate", False, np.full_like(x, np.nan), objective
    yield "infinite_candidate", False, np.full_like(x, np.inf), objective
    yield "wrong_shape", False, x[:-1], objective
    yield "wrong_objective", False, x, -objective
    yield "nonfinite_objective", False, x, np.inf
    for factor, expected in ((.5, True), (2., False)):
        near = np.zeros_like(x)
        near[0] = -factor * 1e-7
        yield f"minimum_tolerance_{factor}", expected, near, float(contract.profit @ near)


def summarize(rows):
    positives = [r for r in rows if r["expected_accept"]]
    negatives = [r for r in rows if not r["expected_accept"]]
    return {
        "cases": len(rows), "positive_controls": len(positives), "invalid_cases": len(negatives),
        "false_acceptances": sum(r["actual_accept"] is True for r in negatives),
        "false_rejections": sum(r["actual_accept"] is False for r in positives),
        "unexpected_errors": sum(r["error"] is not None for r in rows),
        "passed": bool(rows) and all(r["passed"] for r in rows),
    }


def run_challenge(*, seed=20260929, sizes=(2, 8, 32), scales=(.01, 1., 10000.)):
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError("seed must be an unsigned 32-bit integer")
    if not sizes or any(isinstance(n, bool) or not isinstance(n, int) or not 2 <= n <= 1000 for n in sizes):
        raise ValueError("sizes must contain integers between 2 and 1000")
    if not scales or any(not np.isfinite(s) or not .0001 <= s <= 1e6 for s in scales):
        raise ValueError("scales must be finite and between 1e-4 and 1e6")
    protocol = {"schema": "solverpilot.production-challenge-protocol.v1", "seed": seed,
                "sizes": list(sizes), "scales": list(scales), "atol": 1e-7, "rtol": 1e-9}
    rows, blind_spots = [], []
    for index, (n, scale) in enumerate((n, s) for n in sizes for s in scales):
        contract, x, p = fixture(seed + index, n, scale)
        label, digest = f"n{n}-scale{scale}", contract.data_hash
        for name, expected, candidate in formulation_cases(contract, p):
            observe(rows, "formulation", f"{label}/{name}", expected,
                    lambda candidate=candidate: check_production_formulation(contract, candidate).matches,
                    contract_hash=digest)
        for name, expected, candidate, objective in candidate_cases(contract, x):
            def check(candidate=candidate, objective=objective):
                result = check_production_candidate(contract, candidate, objective,
                    atol=protocol["atol"], rtol=protocol["rtol"])
                return result["feasible"] and result["objective_consistent"]
            observe(rows, "candidate", f"{label}/{name}", expected, check, contract_hash=digest)

        # Isolate the maximum bound from resource feasibility: an implementation
        # that checks capacities but forgets product maxima must fail this case.
        headroom = replace(contract, capacity=contract.capacity * 10)
        above_maximum = x * 2.1
        def maximum_check():
            checked = check_production_candidate(headroom, above_maximum,
                float(headroom.profit @ above_maximum), atol=protocol["atol"], rtol=protocol["rtol"])
            return checked["feasible"] and checked["objective_consistent"]
        observe(rows, "candidate", f"{label}/isolated_maximum_overrun", False, maximum_check,
                contract_hash=headroom.data_hash)

        # A wrong source assumption repeated in both contract and model is a
        # deliberate blind-spot demonstration, excluded from detector accuracy.
        relabeled = replace(contract, objective_unit="wrong_but_consistent_unit_label")
        blind_spots.append({"name": f"{label}/unchecked_unit_label", "in_scope": False,
                           "accepted": check_production_formulation(relabeled, p).matches,
                           "reason": "unit labels have no dimensional semantics"})
        wrong_contract = replace(contract, capacity=contract.capacity * 1000)
        wrong_model = replace(p, constraint_upper=p.constraint_upper * 1000)
        blind_spots.append({"name": f"{label}/wrong_input_in_both_paths", "in_scope": False,
                           "accepted": check_production_formulation(wrong_contract, wrong_model).matches,
                           "reason": "no external source of business truth was provided"})

    contract, _, p = fixture(seed, 2, 1.)
    bundle = evidence_bundle(p, solve(p, backend="scipy-highs-ds"))
    def accepted(payload):
        try:
            verify_evidence_bundle(_seal(payload))
        except ValueError:
            return False
        return True
    observe(rows, "evidence", "consistent_original", True, lambda: accepted(bundle))
    mutations = [
        ("optimality_flag", lambda b: b["explanation"]["optimality"].update(
            independently_verified_optimal=not b["explanation"]["optimality"]["independently_verified_optimal"])),
        ("feasibility_flag", lambda b: b["explanation"]["validation"].update(valid=False)),
        ("objective", lambda b: b["explanation"].update(objective=999999.)),
        ("backend", lambda b: b["explanation"].update(backend="different-backend")),
        ("status", lambda b: b["explanation"].update(status="error")),
        ("identity", lambda b: b["explanation"]["runtime"].update(execution_id="different-execution")),
        ("source", lambda b: b["explanation"]["runtime"].update(source_sha256="0" * 64)),
        ("non_boolean_evidence", lambda b: b["explanation"]["optimality"].update(dual_verified="true")),
    ]
    for name, mutate in mutations:
        payload = deepcopy(bundle)
        mutate(payload)
        observe(rows, "evidence", name, False, lambda payload=payload: accepted(payload))
    return {"schema": "solverpilot.production-challenge.v1", "protocol": protocol,
            "protocol_sha256": canonical_sha256(protocol), "source_sha256": source_tree_sha256(),
            "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
            "environment": environment(), "machine": platform.machine(),
            "summary": summarize(rows), "rows": rows, "scope_demonstrations": blind_spots,
            "independent_holdout": False, "human_pilot_completed": False,
            "interpretation": "Seeded synthetic development challenge; no measured competitor superiority, "
                              "general semantic completeness or industrial qualification."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=20260929)
    args = parser.parse_args()
    result = run_challenge(seed=args.seed)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as handle:
        json.dump(result, handle, indent=2, allow_nan=False)
        handle.write("\n")
    print(json.dumps(result["summary"]))
    return 0 if result["summary"]["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
