from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import sparse

from solverpilot import LinearProblem, Session
from solverpilot.session import MutationKind

OUT = Path(__file__).resolve().parent / "results" / "m1-property-checks.json"


def make_problem(seed: int, n: int = 12, m: int = 8) -> LinearProblem:
    rng = np.random.default_rng(seed)
    A = sparse.random(
        m,
        n,
        density=0.25,
        random_state=rng,
        data_rvs=lambda k: rng.uniform(0.2, 2.0, size=k),
        format="csr",
    )
    # Guarantee every row/column gets deterministic structure for mutation tests.
    A = A.tolil()
    for i in range(min(m, n)):
        A[i, i] = 1.0 + i / 10
    A = A.tocsr()
    x0 = rng.uniform(0.1, 0.9, size=n)
    ax = np.asarray(A @ x0).ravel()
    return LinearProblem.from_data(
        A=A,
        c=rng.normal(size=n),
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n),
        constraint_lower=np.full(m, -np.inf),
        constraint_upper=ax + 1.0,
    )


def main():
    checks = 0
    for seed in range(250):
        p = make_problem(seed)

        # Objective-only mutation must preserve structure.
        s = Session(p)
        new_c = np.array(p.c, copy=True)
        new_c[0] += 0.5
        rec = s.update(c=new_c)
        assert rec.kinds == (MutationKind.OBJECTIVE_VECTOR,)
        assert rec.old_structural_hash == rec.new_structural_hash
        checks += 1

        # Same-sparsity matrix value mutation must preserve structural hash.
        s = Session(p)
        A2 = p.A.copy()
        A2.data = np.array(A2.data, copy=True)
        A2.data[0] *= 1.5
        rec = s.update(A=A2)
        assert MutationKind.MATRIX_VALUES_SAME_SPARSITY in rec.kinds
        assert MutationKind.MATRIX_SPARSITY_CHANGED not in rec.kinds
        assert rec.old_structural_hash == rec.new_structural_hash
        checks += 1

        # Removing one explicit nonzero must be structural.
        s = Session(p)
        A3 = p.A.copy().tolil()
        row, col = p.A.nonzero()
        A3[int(row[0]), int(col[0])] = 0.0
        rec = s.update(A=A3.tocsr())
        assert MutationKind.MATRIX_SPARSITY_CHANGED in rec.kinds
        assert rec.old_structural_hash != rec.new_structural_hash
        checks += 1

        # No-op must not advance revision.
        s = Session(p)
        rec = s.update(c=np.array(p.c, copy=True))
        assert rec.is_noop
        assert s.revision == 0
        checks += 1

    payload = {"seeds": 250, "checks": checks, "passed": checks}
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
