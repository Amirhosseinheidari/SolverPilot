"""Small reproducible upgrade measurements; not an industrial performance claim."""

import argparse
from importlib.metadata import version
import json
import platform
from statistics import median
from time import perf_counter
from pathlib import Path
import numpy as np
from scipy import sparse
from solverpilot import LinearProblem, QuadraticProblem, solve, __version__
from solverpilot.backends import OSQPNativeBackend, ScipyHighsLPBackend
from solverpilot.backends.pdlp import PDLPBackend


def measure(problem, backend):
    start = perf_counter()
    r = solve(problem, backend=backend)
    elapsed = perf_counter() - start
    return dict(
        wall_s=elapsed,
        valid=r.validation.valid,
        objective=r.objective,
        independent=r.optimality_evidence.independently_verified_optimal,
        backend_status=r.backend_status,
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    n = 10000
    A = sparse.eye(n, format="csr") + sparse.diags(
        [np.full(n - 1, 0.2)], [1], shape=(n, n), format="csr"
    )
    p = LinearProblem.from_data(
        A=A,
        c=np.ones(n),
        variable_lower=np.zeros(n),
        variable_upper=np.full(n, 10.0),
        constraint_lower=np.ones(n),
        constraint_upper=np.full(n, np.inf),
    )
    lp = {}
    for name, b in [("highs-ds", ScipyHighsLPBackend()), ("pdlp", PDLPBackend(time_limit_s=30))]:
        lp[name] = [measure(p, b) for _ in range(3)]
    n = 300
    qp = {}
    b = OSQPNativeBackend()
    for mode in ("cold", "reused"):
        rows = []
        for i in range(12):
            p = QuadraticProblem.from_data(
                P=sparse.eye(n, format="csr") * 2,
                q=np.full(n, -1.0 - i * 0.01),
                A=sparse.csr_matrix((0, n)),
                variable_lower=np.zeros(n),
                variable_upper=np.full(n, 2.0),
                constraint_lower=[],
                constraint_upper=[],
            )
            rows.append(measure(p, OSQPNativeBackend() if mode == "cold" else b))
        qp[mode] = rows
    payload = dict(
        version=__version__,
        platform=platform.platform(),
        python=platform.python_version(),
        dependencies={k: version(k) for k in ("numpy", "scipy", "osqp", "ortools")},
        protocol="serial; total call includes inspection, build, solve, validation and worker startup; no statistical speed claim",
        lp_10000=lp,
        qp_300=qp,
        median_wall_s={
            **{"lp_" + k: median(r["wall_s"] for r in v) for k, v in lp.items()},
            **{"qp_" + k: median(r["wall_s"] for r in v) for k, v in qp.items()},
        },
    )
    args.output.write_text(json.dumps(payload, indent=2) + "\n")
    if not all(r["valid"] for table in (lp, qp) for rows in table.values() for r in rows):
        raise SystemExit("benchmark candidate validation failed")
    print(json.dumps(payload["median_wall_s"], indent=2))


if __name__ == "__main__":
    main()
