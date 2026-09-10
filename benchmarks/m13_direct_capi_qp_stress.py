from __future__ import annotations

import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from solverpilot.backends import BundledOSQPCAPIBackend, CasadiOSQPBridgeBackend
from solverpilot.runtime import execute
from m9_cross_library_qp import make_case

OUT = Path(__file__).parent / "results" / "m13-direct-capi-qp-stress.json"
SIZES = (5, 20, 60)
FAMILIES = ("interior", "equality", "upper_active", "box_active")
SEEDS = 8


def main() -> None:
    probe = BundledOSQPCAPIBackend()
    if not probe.is_available():
        raise SystemExit("verified bundled OSQP C API unavailable")
    direct_manifest = probe.manifest
    probe.close()

    rows = []
    started = perf_counter()
    for n in SIZES:
        for fi, family in enumerate(FAMILIES):
            for local_seed in range(SEEDS):
                seed = 130000 + n * 100 + fi * 10 + local_seed
                problem, x_star, obj_star = make_case(n, family, seed)

                direct = BundledOSQPCAPIBackend()
                bridge = CasadiOSQPBridgeBackend(enable_primal_warm_start=False)
                t0 = perf_counter(); rd = execute(problem, direct); direct_wall = perf_counter() - t0
                t0 = perf_counter(); rb = execute(problem, bridge); bridge_wall = perf_counter() - t0
                direct.close()

                rows.append({
                    "n": n,
                    "family": family,
                    "seed": seed,
                    "reference_objective": obj_star,
                    "direct": {
                        "status": rd.status.value,
                        "valid": bool(rd.validation and rd.validation.valid),
                        "objective": rd.objective,
                        "objective_abs_error": None if rd.objective is None else float(abs(rd.objective - obj_star)),
                        "x_inf_error": None if rd.x is None else float(np.max(np.abs(rd.x - x_star))),
                        "wall_s": direct_wall,
                        "solver_wall_s": float(rd.raw_statistics["solve_wall_s"]),
                    },
                    "casadi_bridge": {
                        "status": rb.status.value,
                        "valid": bool(rb.validation and rb.validation.valid),
                        "objective": rb.objective,
                        "objective_abs_error": None if rb.objective is None else float(abs(rb.objective - obj_star)),
                        "x_inf_error": None if rb.x is None else float(np.max(np.abs(rb.x - x_star))),
                        "wall_s": bridge_wall,
                    },
                    "direct_vs_bridge_objective_abs_diff": None if rd.objective is None or rb.objective is None else float(abs(rd.objective - rb.objective)),
                })

    payload = {
        "benchmark": "M13 direct OSQP 0.6.3 C API vs CasADi OSQP bridge on constructed-KKT convex QP",
        "claim_scope": "binding/ABI correctness verification; not public OSQP Python package benchmark",
        "instances": len(rows),
        "solver_calls": len(rows) * 2,
        "direct_failures": sum(not r["direct"]["valid"] for r in rows),
        "bridge_failures": sum(not r["casadi_bridge"]["valid"] for r in rows),
        "max_direct_objective_abs_error": max(float(r["direct"]["objective_abs_error"] or 0.0) for r in rows),
        "max_bridge_objective_abs_error": max(float(r["casadi_bridge"]["objective_abs_error"] or 0.0) for r in rows),
        "max_direct_vs_bridge_objective_abs_diff": max(float(r["direct_vs_bridge_objective_abs_diff"] or 0.0) for r in rows),
        "max_direct_x_inf_error": max(float(r["direct"]["x_inf_error"] or 0.0) for r in rows),
        "direct_manifest": {
            "solver_version": direct_manifest.version,
            "library_sha256": direct_manifest.metadata["library_sha256"],
            "header_sha256": direct_manifest.metadata["header_sha256"],
            "bridge_package_version": direct_manifest.metadata["bridge_package_version"],
        },
        "environment": {"python": sys.version, "platform": platform.platform(), "cpu_count": os.cpu_count()},
        "wall_s": perf_counter() - started,
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps({k: payload[k] for k in (
        "instances", "solver_calls", "direct_failures", "bridge_failures",
        "max_direct_objective_abs_error", "max_direct_vs_bridge_objective_abs_diff", "wall_s"
    )}, indent=2))


if __name__ == "__main__":
    main()
