from __future__ import annotations

import argparse
import importlib.util
import json
import pathlib
import platform
import tomllib
from importlib.metadata import PackageNotFoundError, version as dist_version

import numpy as np

import solverpilot
from solverpilot import LinearProblem, QuadraticProblem, solve, solve_production
from solverpilot.cli.backend_health import build_report

ROOT = pathlib.Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
EXPECTED_VERSION = PROJECT["version"]
EXPECTED_NAME = PROJECT["name"]


def backend(report, name):
    return next((x for x in report["backends"] if x["backend"] == name), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--require-extra",
        choices=["highs", "osqp", "scip", "nlopt", "casadi", "conic", "nlp", "minlp", "cp", "clarabel"],
    )
    ap.add_argument("--forbid-source-root")
    ns = ap.parse_args()
    extra_map = {
        "highs": "highspy-native",
        "osqp": "osqp-native",
        "scip": "pyscipopt-native",
        "nlopt": "nlopt-slsqp-native",
        "casadi": "casadi-highs-bridge",
    }

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
        dist = dist_version(EXPECTED_NAME)
    except PackageNotFoundError:
        dist = None
    import_path = str(pathlib.Path(solverpilot.__file__).resolve())
    legacy_spec = importlib.util.find_spec("optimind")

    extended = None
    if ns.require_extra == "scip":
        from solverpilot.globalopt import SCIPGlobalBackend
        from solverpilot.runtime.catalog import verify_specialized_backend
        b = SCIPGlobalBackend()
        extended = {"kind": "global", "capabilities": b.capabilities(),
                    "passed": verify_specialized_backend(b)}
    elif ns.require_extra == "clarabel":
        from solverpilot.conic import ClarabelBackend
        from solverpilot.conic.clarabel_backend import _runtime_conformance
        b = ClarabelBackend()
        extended = {"kind": "clarabel", "binding_version": b.binding_version,
                    "passed": b.is_available() and _runtime_conformance(b.binding_version)}
    elif ns.require_extra == "conic":
        from importlib.metadata import version as package_version
        from solverpilot.model import Model
        from solverpilot.conic import ConicProblem, ConeKind
        m = Model("release-soc")
        x = m.variable(1)
        t = m.variable(lower=0.0)
        m.soc(t, x)
        m.minimize(t)
        compiled = m.compile(use_cache=False)
        casadi_version = package_version("casadi")
        extended = {
            "kind": "conic",
            "casadi_version": casadi_version,
            "compiled_ir": type(compiled.execution_ir).__name__,
            "soc_kind": compiled.execution_ir.cones[0].kind.value,
            "passed": isinstance(compiled.execution_ir, ConicProblem)
            and compiled.execution_ir.cones[0].kind is ConeKind.SECOND_ORDER
            and casadi_version == "3.7.2",
        }
    elif ns.require_extra == "nlp":
        from importlib.metadata import version as package_version
        from solverpilot.model import Model
        from solverpilot.nlp import CasadiIpoptBackend
        b = CasadiIpoptBackend()
        m = Model("release-nlp")
        x = m.variable(lower=-3.0, upper=3.0)
        m.minimize((x - 1.25) ** 4 + 0.2 * (x - 1.25) ** 2)
        r = m.solve(x0=[0.0]) if b.is_available() else None
        extended = {
            "kind": "nlp",
            "casadi_version": package_version("casadi"),
            "available": b.is_available(),
            "valid": bool(r and r.validation.valid),
            "local_optimal_candidate": bool(r and r.local_optimal_candidate),
            "globally_proven": bool(r and r.globally_proven),
        }
        extended["passed"] = bool(extended["casadi_version"] == "3.7.2" and extended["available"] and extended["valid"] and extended["local_optimal_candidate"] and not extended["globally_proven"])
    elif ns.require_extra == "minlp":
        from importlib.metadata import version as package_version
        from solverpilot.model import Model
        from solverpilot.nlp import CasadiIpoptBackend
        b = CasadiIpoptBackend()
        m = Model("release-minlp")
        x = m.variable(lower=0.0, upper=3.0)
        z = m.binary()
        m.minimize((x - 2.0) ** 2 + 0.2 * (1 - z))
        r = m.compile().solve() if b.is_available() else None
        extended = {
            "kind": "minlp",
            "casadi_version": package_version("casadi"),
            "nlp_available": b.is_available(),
            "status": getattr(r, "status", None),
            "globally_proven": bool(r and r.globally_proven),
            "proof_scope": getattr(r, "proof_scope", None),
            "independently_verified_global": bool(r and r.independently_verified_global),
            "objective": getattr(r, "objective", None),
        }
        extended["passed"] = bool(
            extended["casadi_version"] == "3.7.2"
            and extended["nlp_available"]
            and extended["globally_proven"]
            and extended["proof_scope"] == "solver_certified_under_convexity_assumptions"
            and not extended["independently_verified_global"]
            and extended["status"] == "proven_optimal"
            and abs(float(extended["objective"])) < 1e-5
        )
    elif ns.require_extra == "cp":
        from importlib.metadata import version as package_version
        from solverpilot.cp import CPModel, ORToolsCPSATBackend, VERIFIED_ORTOOLS_VERSION
        b = ORToolsCPSATBackend()
        m = CPModel("release-cp")
        x = m.bool_var("x")
        y = m.bool_var("y")
        m.add_exactly_one([x, y])
        m.maximize(3 * x + 2 * y)
        r = m.solve(b) if b.is_available() else None
        extended = {
            "kind": "cp",
            "ortools_version": package_version("ortools"),
            "verified_version": VERIFIED_ORTOOLS_VERSION,
            "available": b.is_available(),
            "valid": bool(r and r.validation.valid),
            "optimality_proven": bool(r and r.optimality_proven),
            "objective": getattr(r, "objective", None),
        }
        extended["passed"] = bool(extended["ortools_version"] == VERIFIED_ORTOOLS_VERSION and extended["available"] and extended["valid"] and extended["optimality_proven"] and extended["objective"] == 3)
        from solverpilot.backends.pdlp import PDLPBackend
        pdlp_result = solve(lp, backend=PDLPBackend(time_limit_s=20))
        extended["pdlp_valid"] = bool(pdlp_result.validation.valid and abs(pdlp_result.objective-1)<1e-6)
        extended["passed"] = extended["passed"] and extended["pdlp_valid"]

    payload = {
        "schema": "solverpilot.release.smoke.v2",
        "expected_version": EXPECTED_VERSION,
        "version": solverpilot.__version__,
        "distribution_version": dist,
        "python": platform.python_version(),
        "platform": platform.platform(),
        "import_path": import_path,
        "legacy_optimind_namespace_present": legacy_spec is not None,
        "lp": {"status": lr.status.value, "valid": bool(lr.validation and lr.validation.valid)},
        "milp": {"status": mr.status.value, "valid": bool(mr.validation and mr.validation.valid)},
        "qp": {"status": qr.status.value, "valid": bool(qr.validation and qr.validation.valid)},
        "production": {"status": pr.status.value, "valid": bool(pr.validation and pr.validation.valid), "backend": decision.plan.selected_backend, "performance_ranking": decision.auto_performance_ranking_enabled},
        "required_backend": req,
        "extended_extra": extended,
    }
    checks = [
        solverpilot.__version__ == EXPECTED_VERSION,
        dist == EXPECTED_VERSION,
        legacy_spec is None,
        payload["lp"]["valid"], payload["milp"]["valid"], payload["qp"]["valid"], payload["production"]["valid"],
        decision.plan.selected_backend == "scipy-highs-ds",
        decision.auto_performance_ranking_enabled is False,
    ]
    if ns.forbid_source_root:
        checks.append(not import_path.startswith(str(pathlib.Path(ns.forbid_source_root).resolve())))
    if req_name:
        checks.append(req is not None and req.get("status") == "healthy")
    if extended is not None:
        checks.append(bool(extended.get("passed")))
    payload["passed"] = bool(all(checks))
    print(json.dumps(payload, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
