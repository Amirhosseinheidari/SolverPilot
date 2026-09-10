from __future__ import annotations

import numpy as np
import pytest

import solverpilot as om

pytestmark = pytest.mark.skipif(not om.CasadiSuperSCSBackend().is_available(), reason='CasADi SuperSCS plugin unavailable')


def test_superscs_soc_345():
    m=om.Model(); x=m.variable(2,lower=[3,4],upper=[3,4]); t=m.variable(lower=0,upper=10); m.soc(t,x); m.minimize(t)
    c=m.compile(use_cache=False); r=c.solve(backend=om.CasadiSuperSCSBackend(eps=1e-9,max_iter=100000))
    assert r.validation.valid
    assert abs(r.objective_reported-5.0) <= 2e-6
    assert c.validate_original(m,r.x,atol=2e-6).valid


def test_superscs_rotated_soc_analytic():
    m=om.Model(); z=m.variable(1,lower=[1],upper=[1]); u=m.variable(lower=0,upper=5); v=m.variable(lower=0,upper=5); m.rotated_soc(u,v,z); m.minimize(u+v)
    c=m.compile(use_cache=False); r=c.solve(backend=om.CasadiSuperSCSBackend(eps=1e-9,max_iter=100000))
    assert r.validation.valid
    assert abs(r.objective_reported-np.sqrt(2.0)) <= 5e-6
    assert c.validate_original(m,r.x,atol=5e-6).valid


def test_superscs_psd_fails_closed_after_negative_conformance():
    m=om.Model(); t=m.variable(lower=0,upper=5); M=t*m.constant(np.eye(2))+m.constant([[0.,1.],[1.,0.]])
    m.psd(M); m.minimize(t)
    c=m.compile(use_cache=False)
    with pytest.raises(RuntimeError, match='constraint.psd: capability is unverified'):
        c.solve(backend=om.CasadiSuperSCSBackend())

def test_superscs_mixed_linear_and_soc():
    m=om.Model(); x=m.variable(2,lower=-10,upper=10); t=m.variable(lower=0,upper=10)
    m.add(x[0] == 1.0); m.add(x[1] == 2.0); m.soc(t,x); m.minimize(t)
    r=m.compile(use_cache=False).solve(backend=om.CasadiSuperSCSBackend(eps=1e-9,max_iter=100000))
    assert r.validation.valid
    assert abs(r.objective_reported-np.sqrt(5.0)) <= 1e-5


def test_random_soc_instances_against_analytic_norm():
    rng=np.random.default_rng(6106)
    backend=om.CasadiSuperSCSBackend(eps=1e-8,max_iter=100000)
    for _ in range(30):
        vec=rng.normal(size=3)
        m=om.Model(); x=m.variable(3,lower=vec,upper=vec); t=m.variable(lower=0,upper=20); m.soc(t,x); m.minimize(t)
        r=m.compile(use_cache=False).solve(backend=backend)
        assert r.validation.valid
        assert abs(r.objective_reported-np.linalg.norm(vec)) <= 2e-4



def test_quadratic_soc_ir_compiles_but_superscs_fails_closed():
    m=om.Model(); x=m.variable(2,lower=-10,upper=10); a=np.array([3.,4.]); m.soc(2.0,x); m.minimize(0.5*((x-m.constant(a))**2).sum())
    c=m.compile(use_cache=False)
    assert c.execution_ir.P.nnz > 0
    expected=np.array([1.2,1.6])
    assert om.validate_conic_solution(c.execution_ir, expected, atol=1e-8, rtol=1e-8).valid
    with pytest.raises(RuntimeError, match='problem.conic_quadratic: capability is unverified'):
        c.solve(backend=om.CasadiSuperSCSBackend())
