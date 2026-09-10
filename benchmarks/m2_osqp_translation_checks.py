from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from solverpilot import QuadraticProblem, Session
from solverpilot.backends.osqp_native import _osqp_data

OUT = Path(__file__).resolve().parent / "results" / "m2-osqp-translation-checks.json"


def make_qp(seed: int, n: int = 18, m: int = 12):
    rng = np.random.default_rng(seed)
    diag = rng.uniform(0.5, 4.0, size=n)
    P = sparse.diags(diag, format="csr")
    A = sparse.random(
        m, n, density=0.2, random_state=rng,
        data_rvs=lambda k: rng.normal(size=k), format="csr"
    )
    x0 = rng.uniform(-0.5, 0.5, size=n)
    ax = np.asarray(A @ x0).ravel()
    return QuadraticProblem.from_data(
        P=P, A=A, q=rng.normal(size=n),
        variable_lower=np.full(n, -2.0), variable_upper=np.full(n, 2.0),
        constraint_lower=ax - 1.0, constraint_upper=ax + 1.0,
    )


def pattern(M):
    M = M.tocsc()
    return (M.shape, M.indptr.tobytes(), M.indices.tobytes())


def main():
    checks = 0
    for seed in range(200):
        p = make_qp(70000 + seed)
        d0 = _osqp_data(p)
        s = Session(p)

        # Vector-only update: all OSQP matrix patterns must remain identical.
        q = np.array(p.linear.c, copy=True)
        q[0] += 0.1
        s.update(q=q)
        d1 = _osqp_data(s.problem)
        assert pattern(d0.P) == pattern(d1.P)
        assert pattern(d0.A) == pattern(d1.A)
        checks += 2

        # Same-sparsity A value update must retain the exact CSC pattern needed by OSQP update.
        A2 = s.problem.linear.A.copy()
        if A2.nnz:
            A2.data = np.array(A2.data, copy=True)
            A2.data[0] *= 1.1
            s.update(A=A2)
            d2 = _osqp_data(s.problem)
            assert pattern(d1.A) == pattern(d2.A)
            assert d1.A.data.size == d2.A.data.size
            checks += 2

        # Same-sparsity Hessian value update must retain upper-triangle update ordering.
        P2 = s.problem.P.copy()
        if P2.nnz:
            P2.data = np.array(P2.data, copy=True)
            P2.data[0] *= 1.05
            s.update(P=P2)
            d3 = _osqp_data(s.problem)
            assert pattern(d2.P) == pattern(d3.P)
            assert d2.P.data.size == d3.P.data.size
            checks += 2

    payload = {"seeds": 200, "checks": checks, "passed": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
