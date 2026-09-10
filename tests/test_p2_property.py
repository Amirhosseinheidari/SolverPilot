from __future__ import annotations

import numpy as np

from solverpilot.model import Model
from solverpilot.problem import LinearProblem, QuadraticProblem


def _eq_sparse(a,b,atol=1e-12):
    np.testing.assert_allclose(a.toarray(),b.toarray(),atol=atol,rtol=0)


def _eq_problem(a,b):
    if isinstance(a,QuadraticProblem):
        assert isinstance(b,QuadraticProblem); _eq_sparse(a.P,b.P); a=a.linear; b=b.linear
    else:
        assert isinstance(a,LinearProblem) and isinstance(b,LinearProblem)
    _eq_sparse(a.A,b.A); np.testing.assert_allclose(a.c,b.c,atol=1e-12,rtol=0)
    np.testing.assert_allclose(a.variable_lower,b.variable_lower,atol=0,rtol=0)
    np.testing.assert_allclose(a.variable_upper,b.variable_upper,atol=0,rtol=0)
    np.testing.assert_allclose(a.constraint_lower,b.constraint_lower,atol=1e-12,rtol=0)
    np.testing.assert_allclose(a.constraint_upper,b.constraint_upper,atol=1e-12,rtol=0)
    assert a.objective_sense==b.objective_sense
    assert a.objective_offset==b.objective_offset


def test_cached_vs_full_linear_recompile_400_random_updates():
    rng=np.random.default_rng(2026090602)
    m=Model(); x=m.variable(8,lower=-3,upper=4)
    A=m.parameter((6,8),value=rng.normal(size=(6,8)))
    b=m.parameter(6,value=rng.normal(size=6))
    q=m.parameter(8,value=rng.normal(size=8))
    m.add(A@x<=b); m.minimize(q@x)
    first=m.compile(); sem=first.semantic_hash; plan=first.compilation_hash
    for _ in range(400):
        A.value=rng.normal(size=(6,8)); b.value=rng.normal(size=6); q.value=rng.normal(size=8)
        cached=m.compile(); full=m.compile(use_cache=False)
        assert cached.semantic_hash==sem; assert cached.compilation_hash==plan
        _eq_problem(cached.execution_ir,full.execution_ir)


def test_cached_vs_full_partial_dependency_updates_300_cases():
    rng=np.random.default_rng(8821)
    m=Model(); x=m.variable(12,lower=-5,upper=5)
    ps=[m.parameter(value=float(i+1)) for i in range(12)]
    for i,p in enumerate(ps): m.add(x[i]<=p)
    q=m.parameter(12,value=np.ones(12)); m.minimize(q@x); m.compile()
    for _ in range(300):
        idx=int(rng.integers(0,12)); ps[idx].value=float(rng.normal())
        if rng.random()<0.2: q.value=rng.normal(size=12)
        cached=m.compile(); full=m.compile(use_cache=False); _eq_problem(cached.execution_ir,full.execution_ir)
        assert cached.compilation_report.recompiled_execution_rows in {1,0}


def test_cached_vs_full_convex_diagonal_qp_180_updates():
    rng=np.random.default_rng(2219)
    n=6; m=Model(); x=m.variable(n,lower=-4,upper=4)
    d=m.parameter(n,value=np.ones(n),sign="nonnegative")
    q=m.parameter(n,value=np.zeros(n))
    expr=q@x
    for i in range(n): expr=expr+0.5*d[i]*x[i]*x[i]
    m.minimize(expr); first=m.compile(); plan=first.compilation_hash
    for _ in range(180):
        d.value=rng.uniform(0,4,size=n); q.value=rng.normal(size=n)
        cached=m.compile(); full=m.compile(use_cache=False)
        assert cached.compilation_hash==plan; _eq_problem(cached.execution_ir,full.execution_ir)
