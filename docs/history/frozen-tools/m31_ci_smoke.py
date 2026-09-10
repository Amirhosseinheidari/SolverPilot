from __future__ import annotations

import argparse
import json
import platform
import sys
from importlib.metadata import PackageNotFoundError, version as dist_version

import numpy as np

import solverpilot
from solverpilot import LinearProblem, QuadraticProblem, solve, solve_production
from solverpilot.cli.backend_health import build_report


def _lp() -> LinearProblem:
    return LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )


def _milp() -> LinearProblem:
    return LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[1.0],
        constraint_upper=[1.0],
        domains=["binary", "binary"],
    )


def _qp() -> QuadraticProblem:
    return QuadraticProblem.from_data(
        P=[[2.0]],
        A=[[1.0]],
        q=[-2.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[0.0],
        constraint_upper=[2.0],
    )


def _backend_entry(report: dict[str, object], name: str) -> dict[str, object] | None:
    for entry in report["backends"]:  # type: ignore[index]
        if entry["backend"] == name:  # type: ignore[index]
            return entry  # type: ignore[return-value]
    return None


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--require-backend")
    parser.add_argument("--require-extra", choices=["highs", "osqp", "scip", "nlopt", "casadi"])
    args = parser.parse_args()

    lp = solve(_lp(), backend="scipy-highs-ds")
    milp = solve(_milp(), backend="scipy-highs-bridge")
    qp = solve(_qp(), backend="scipy-slsqp-qp-bridge")
    prod_result, prod_decision = solve_production(_lp())

    health = build_report()
    extra_map = {"highs": "highspy-native", "osqp": "osqp-native", "scip": "pyscipopt-native", "nlopt": "nlopt-slsqp-native", "casadi": "casadi-highs-bridge"}
    required_name = args.require_backend or (extra_map.get(args.require_extra) if args.require_extra else None)
    required = _backend_entry(health, required_name) if required_name else None

    try:
        installed = dist_version("solverpilot")
    except PackageNotFoundError:
        installed = None

    payload = {
        "schema": "solverpilot.m31.ci_smoke.v1",
        "python": platform.python_version(),
        "platform": platform.platform(),
        "solverpilot_version": solverpilot.__version__,
        "distribution_version": installed,
        "lp": {"status": lp.status.value, "valid": bool(lp.validation and lp.validation.valid)},
        "milp": {"status": milp.status.value, "valid": bool(milp.validation and milp.validation.valid)},
        "qp": {"status": qp.status.value, "valid": bool(qp.validation and qp.validation.valid)},
        "production": {
            "status": prod_result.status.value,
            "valid": bool(prod_result.validation and prod_result.validation.valid),
            "backend": prod_decision.plan.selected_backend,
            "performance_ranking": prod_decision.auto_performance_ranking_enabled,
        },
        "required_backend": required,
    }
    print(json.dumps(payload, sort_keys=True))

    checks = [
        solverpilot.__version__ == "0.0.36rc1",
        installed in {None, "0.0.36rc1"},
        bool(lp.validation and lp.validation.valid),
        bool(milp.validation and milp.validation.valid),
        bool(qp.validation and qp.validation.valid),
        bool(prod_result.validation and prod_result.validation.valid),
        prod_decision.plan.selected_backend == "scipy-highs-ds",
        prod_decision.auto_performance_ranking_enabled is False,
    ]
    if required_name:
        checks.append(required is not None and required.get("status") == "healthy")
    return 0 if all(checks) else 2


if __name__ == "__main__":
    raise SystemExit(main())
