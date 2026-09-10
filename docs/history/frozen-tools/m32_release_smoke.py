from __future__ import annotations

import argparse
import json
import pathlib
import platform
from importlib.metadata import PackageNotFoundError, version as dist_version

import numpy as np

import solverpilot
from solverpilot import LinearProblem, QuadraticProblem, solve, solve_production
from solverpilot.cli.backend_health import build_report

EXPECTED_VERSION = "0.0.36rc2"


def backend(report, name):
    return next((x for x in report["backends"] if x["backend"] == name), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--require-extra", choices=["highs", "osqp", "scip", "nlopt", "casadi"])
    ap.add_argument("--forbid-source-root")
    ns = ap.parse_args()
    extra_map = {"highs": "highspy-native", "osqp": "osqp-native", "scip": "pyscipopt-native", "nlopt": "nlopt-slsqp-native", "casadi": "casadi-highs-bridge"}

    lp = LinearProblem.from_data(A=[[1.,1.]], c=[1.,2.], variable_lower=[0.,0.], variable_upper=[1.,1.], constraint_lower=[1.], constraint_upper=[np.inf])
    milp = LinearProblem.from_data(A=[[1.,1.]], c=[1.,2.], variable_lower=[0.,0.], variable_upper=[1.,1.], constraint_lower=[1.], constraint_upper=[1.], domains=["binary","binary"])
    qp = QuadraticProblem.from_data(P=[[2.]], A=[[1.]], q=[-2.], variable_lower=[0.], variable_upper=[2.], constraint_lower=[0.], constraint_upper=[2.])
    lr = solve(lp, backend="scipy-highs-ds")
    mr = solve(milp, backend="scipy-highs-bridge")
    qr = solve(qp, backend="scipy-slsqp-qp-bridge")
    pr, decision = solve_production(lp)
    health = build_report()
    req_name = extra_map.get(ns.require_extra) if ns.require_extra else None
    req = backend(health, req_name) if req_name else None
    try:
        dist = dist_version("solverpilot")
    except PackageNotFoundError:
        dist = None
    import_path = str(pathlib.Path(solverpilot.__file__).resolve())
    payload = {
        "schema": "solverpilot.m32.release_smoke.v1",
        "version": solverpilot.__version__, "distribution_version": dist,
        "python": platform.python_version(), "platform": platform.platform(), "import_path": import_path,
        "lp": {"status": lr.status.value, "valid": bool(lr.validation and lr.validation.valid)},
        "milp": {"status": mr.status.value, "valid": bool(mr.validation and mr.validation.valid)},
        "qp": {"status": qr.status.value, "valid": bool(qr.validation and qr.validation.valid)},
        "production": {"status": pr.status.value, "valid": bool(pr.validation and pr.validation.valid), "backend": decision.plan.selected_backend, "performance_ranking": decision.auto_performance_ranking_enabled},
        "required_backend": req,
    }
    checks = [
        solverpilot.__version__ == EXPECTED_VERSION, dist == EXPECTED_VERSION,
        payload["lp"]["valid"], payload["milp"]["valid"], payload["qp"]["valid"], payload["production"]["valid"],
        decision.plan.selected_backend == "scipy-highs-ds", decision.auto_performance_ranking_enabled is False,
    ]
    if ns.forbid_source_root:
        checks.append(not import_path.startswith(str(pathlib.Path(ns.forbid_source_root).resolve())))
    if req_name:
        checks.append(req is not None and req.get("status") == "healthy")
    payload["passed"] = bool(all(checks))
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
