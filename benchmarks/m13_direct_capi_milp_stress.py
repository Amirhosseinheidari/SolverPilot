from __future__ import annotations

import itertools
import json
import os
import platform
import sys
from pathlib import Path
from time import perf_counter

import numpy as np

from solverpilot import LinearProblem, ObjectiveSense, VariableDomain
from solverpilot.backends import BundledHighsCAPIBackend, CasadiCBCBridgeBackend, ScipyHighsBackend
from solverpilot.runtime import execute

OUT = Path(__file__).parent / "results" / "m13-direct-capi-milp-stress.json"
CASES = 100


def brute(problem: LinearProblem):
    best = None
    for bits in itertools.product((0.0, 1.0), repeat=problem.n_variables):
        x = np.asarray(bits)
        ax = problem.A @ x
        if np.any(ax < problem.constraint_lower - 1e-10) or np.any(ax > problem.constraint_upper + 1e-10):
            continue
        value = float(problem.c @ x + problem.objective_offset)
        if best is None or value > best:
            best = value
    return best


def make(seed: int) -> LinearProblem:
    rng = np.random.default_rng(seed)
    n = int(rng.integers(6, 13)); m = int(rng.integers(3, 8))
    A = rng.integers(0, 8, size=(m, n)).astype(float)
    c = rng.integers(1, 30, size=n).astype(float)
    caps = np.maximum(1, np.floor(A.sum(axis=1) * rng.uniform(0.25, 0.55, size=m)))
    return LinearProblem.from_data(
        A=A, c=c,
        variable_lower=np.zeros(n), variable_upper=np.ones(n),
        constraint_lower=np.full(m, -np.inf), constraint_upper=caps,
        domains=[VariableDomain.BINARY] * n,
        objective_sense=ObjectiveSense.MAXIMIZE,
        metadata={"seed": seed},
    )


def main() -> None:
    probe = BundledHighsCAPIBackend()
    if not probe.is_available():
        raise SystemExit("verified bundled HiGHS C API unavailable")
    manifest = probe.manifest; probe.close()
    rows=[]; started=perf_counter()
    for i in range(CASES):
        seed=140000+i; problem=make(seed); ref=brute(problem)
        backends={
            "direct_highs_capi": BundledHighsCAPIBackend(),
            "scipy_highs": ScipyHighsBackend(),
            "cbc": CasadiCBCBridgeBackend(),
        }
        solvers={}
        for name, backend in backends.items():
            t0=perf_counter(); r=execute(problem,backend); wall=perf_counter()-t0
            solvers[name]={
                "status":r.status.value,
                "valid":bool(r.validation and r.validation.valid),
                "objective":r.objective,
                "objective_abs_error":None if r.objective is None or ref is None else float(abs(r.objective-ref)),
                "wall_s":wall,
            }
            if hasattr(backend,"close"): backend.close()
        rows.append({"seed":seed,"n":problem.n_variables,"m":problem.n_constraints,"reference_objective":ref,"solvers":solvers})
    names=("direct_highs_capi","scipy_highs","cbc")
    payload={
        "benchmark":"M13 direct HiGHS C API MILP exact brute-force cross-check",
        "claim_scope":"direct C-API binding correctness + independent CBC/brute-force reference",
        "instances":len(rows),
        "solver_calls":len(rows)*len(names),
        "failures":{n:sum(not r["solvers"][n]["valid"] for r in rows) for n in names},
        "max_objective_abs_error":{n:max(float(r["solvers"][n]["objective_abs_error"] or 0.0) for r in rows) for n in names},
        "direct_manifest":{
            "solver_version":manifest.version,
            "library_sha256":manifest.metadata["library_sha256"],
            "header_sha256":manifest.metadata["header_sha256"],
            "bridge_package_version":manifest.metadata["bridge_package_version"],
        },
        "environment":{"python":sys.version,"platform":platform.platform(),"cpu_count":os.cpu_count()},
        "wall_s":perf_counter()-started,
        "rows":rows,
    }
    OUT.parent.mkdir(parents=True,exist_ok=True); OUT.write_text(json.dumps(payload,indent=2),encoding="utf-8")
    print(json.dumps({k:payload[k] for k in ("instances","solver_calls","failures","max_objective_abs_error","wall_s")},indent=2))


if __name__=="__main__": main()
