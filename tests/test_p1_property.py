from __future__ import annotations

import numpy as np
from scipy.optimize import linprog

from solverpilot.model import Model
from solverpilot.problem import LinearProblem


def test_random_affine_compile_equivalence_300_cases():
    rng = np.random.default_rng(20260906)
    for _ in range(300):
        n = int(rng.integers(1, 9))
        mrows = int(rng.integers(0, 9))
        A = rng.normal(size=(mrows, n))
        b = rng.normal(size=mrows)
        c = rng.normal(size=n)
        lo = rng.uniform(-3, 0, size=n)
        hi = rng.uniform(0.1, 4, size=n)
        model = Model()
        x = model.variable(n, lower=lo, upper=hi)
        if mrows:
            model.add(A @ x <= b)
        model.minimize(c @ x + 1.25)
        p = model.compile().execution_ir
        assert isinstance(p, LinearProblem)
        np.testing.assert_allclose(p.A.toarray(), A, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.c, c, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.constraint_upper, b, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.variable_lower, lo, rtol=0, atol=1e-12)
        np.testing.assert_allclose(p.variable_upper, hi, rtol=0, atol=1e-12)
        assert p.objective_offset == 1.25


def test_random_parameter_recompile_equivalence_250_updates():
    rng = np.random.default_rng(7731)
    model = Model(); x = model.variable(5, lower=-2, upper=3)
    A = model.parameter((4,5), value=rng.normal(size=(4,5)))
    b = model.parameter(4, value=rng.normal(size=4))
    q = model.parameter(5, value=rng.normal(size=5))
    model.add(A @ x <= b); model.minimize(q @ x)
    semantic = model.semantic_hash
    for _ in range(250):
        av = rng.normal(size=(4,5)); bv=rng.normal(size=4); qv=rng.normal(size=5)
        A.value=av; b.value=bv; q.value=qv
        compiled=model.compile().execution_ir
        assert model.semantic_hash == semantic
        got = compiled.A.toarray()
        # Function-in-set normalization may represent A x <= b as either
        # A x <= b or equivalently -A x >= -b. Check semantic rows.
        for i in range(4):
            if np.allclose(got[i], av[i], atol=1e-12):
                assert np.isneginf(compiled.constraint_lower[i])
                np.testing.assert_allclose([compiled.constraint_upper[i]], [bv[i]], atol=1e-12)
            else:
                np.testing.assert_allclose(got[i], -av[i], atol=1e-12)
                np.testing.assert_allclose([compiled.constraint_lower[i]], [-bv[i]], atol=1e-12)
                assert np.isposinf(compiled.constraint_upper[i])
        np.testing.assert_allclose(compiled.c, qv, atol=1e-12)


def test_random_convex_diagonal_qp_compile_120_cases():
    rng=np.random.default_rng(44)
    for _ in range(120):
        n=int(rng.integers(1,8))
        d=rng.uniform(0.1,4,size=n)
        q=rng.normal(size=n)
        model=Model(); x=model.variable(n, lower=-5, upper=5)
        expr=q @ x
        for i in range(n):
            expr = expr + 0.5*d[i]*x[i]*x[i]
        model.minimize(expr)
        p=model.compile().execution_ir
        np.testing.assert_allclose(p.P.toarray(), np.diag(d), atol=1e-12)
        np.testing.assert_allclose(p.linear.c, q, atol=1e-12)
