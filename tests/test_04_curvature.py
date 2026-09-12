import numpy as np
import pytest
from scipy import sparse
from solverpilot.problem import QuadraticProblem
from solverpilot.problem.curvature import certify_psd
from solverpilot.validate import verify_optimality


def make(P, verify=True):
    n = len(P)
    return QuadraticProblem.from_data(
        P=P,
        A=np.empty((0, n)),
        q=np.zeros(n),
        variable_lower=np.full(n, -1e10),
        variable_upper=np.full(n, 1e10),
        constraint_lower=[],
        constraint_upper=[],
        verify_convexity=verify,
    )


@pytest.mark.parametrize("v", [-1e-14, -1e-12, -1e-10])
def test_negative_curvature_never_enters_convex_runtime(v):
    with pytest.raises(ValueError, match="positive semidefinite"):
        make([[v]])


def test_near_psd_and_singular_boundary():
    with pytest.raises(ValueError, match="positive semidefinite"):
        make([[1.0, 1.0 + 2**-52], [1.0 + 2**-52, 1.0]])
    p = make(np.ones((3, 3)))
    assert certify_psd(p.P).certified
    assert verify_optimality(p, np.zeros(3), np.zeros(3)).verified


def test_exact_congruences_and_budget():
    rng = np.random.default_rng(20260912)
    for n in [1, 2, 4, 8]:
        for _ in range(20):
            L = np.tril(rng.integers(-3, 4, size=(n, n)), -1) + np.eye(n, dtype=int)
            for sign in [0, 1, -1]:
                d = rng.integers(1, 5, size=n)
                d[0] = sign
                check = certify_psd(L @ np.diag(d) @ L.T)
                assert check.certified == (sign >= 0)
                if sign < 0:
                    assert check.status == "not_psd"
    assert certify_psd(sparse.eye(10000)).certified
    assert certify_psd(np.ones((33, 33))).status == "unknown"


def test_large_numerically_psd_candidate_not_independently_promoted():
    P = np.ones((33, 33)) + np.eye(33)
    p = make(P)
    check = verify_optimality(p, np.zeros(33), np.zeros(33))
    assert not check.verified
    assert "PSD evidence" in check.reason
