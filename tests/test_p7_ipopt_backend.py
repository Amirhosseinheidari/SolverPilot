import numpy as np
import pytest

pytest.importorskip("casadi", reason="optional CasADi extra not installed in core CI")
from solverpilot.model import Model, sin, exp
from solverpilot.nlp import CasadiIpoptBackend, NLPDerivativeEngine
from solverpilot.capabilities.v2 import CapabilityKey, compatible_v2, requirements_v2_for


def test_ipopt_backend_available_and_capability_verified():
    b=CasadiIpoptBackend(); assert b.is_available(); man=b.capability_manifest_v2
    assert man.claim(CapabilityKey.PROBLEM_NLP).runtime_verified(backend_version=man.backend_version,binding_version=man.binding_version,adapter_version=man.adapter_version)


def test_capability_requirements_accept_verified_ipopt_bridge():
    m=Model(); x=m.variable(lower=-3,upper=3); m.minimize((x-1)**4 + 0.1*sin(x)); p=m.compile().execution_ir
    b=CasadiIpoptBackend(); ok,checks=compatible_v2(b.capability_manifest_v2,requirements_v2_for(p),allow_safe_emulation=True,require_verified=True)
    assert ok,checks


def test_unconstrained_nlp_local_candidate_is_not_global_claim():
    m=Model(); x=m.variable(lower=-4,upper=4); m.minimize((x-1.25)**4 + 0.2*(x-1.25)**2)
    r=m.solve(x0=[0.0]); assert r.validation.valid; assert r.local_optimal_candidate; assert not r.globally_proven; assert r.x[0]==pytest.approx(1.25,abs=2e-5)


def test_constrained_nlp_matches_analytic_solution():
    m=Model(); x=m.variable(2,lower=[-3,-3],upper=[3,3]); m.minimize((x[0]-2)**2+(x[1]+1)**2); m.add(x[0]+x[1] <= 0)
    from solverpilot.nlp import compile_nlp_model
    r=CasadiIpoptBackend().solve(compile_nlp_model(m).execution_ir,x0=[0,0]); assert r.validation.valid; assert r.local_optimal_candidate; assert not r.globally_proven
    assert np.allclose(r.x,[1.5,-1.5],atol=3e-5); assert r.validation.max_constraint_violation<=1e-7


def test_nonlinear_constraint_solution_and_original_validation():
    m=Model(); x=m.variable(2,lower=[-2,-2],upper=[2,2]); m.minimize((x[0]-1)**2+(x[1]-1)**2); m.add(x[0]**2+x[1]**2 <= 1)
    c=m.compile(); r=c.solve(x0=[0.2,0.2]); assert r.validation.valid; orig=c.validate_original(m,r.x,atol=2e-7); assert orig.valid
    target=1/np.sqrt(2); assert np.allclose(r.x,[target,target],atol=5e-5)


def test_backend_returns_multipliers_without_global_proof():
    m=Model(); x=m.variable(lower=-3,upper=3); m.minimize((x-2)**2); m.add(exp(x) <= np.e)
    r=m.solve(x0=[0]); assert r.validation.valid; assert r.multipliers_g is not None; assert not r.globally_proven

def test_local_candidate_requires_recomputed_kkt_stationarity():
    m=Model(); x=m.variable(2,lower=[0,0],upper=[2,2]); m.minimize((x[0]-1)**2+(x[1]-1)**2); m.add(x[0]**2+x[1]**2<=1)
    from solverpilot.nlp import compile_nlp_model
    r=CasadiIpoptBackend().solve(compile_nlp_model(m).execution_ir,x0=[0.2,0.3])
    assert r.local_optimal_candidate; assert r.kkt_stationarity_inf is not None; assert r.kkt_stationarity_inf < 1e-6; assert not r.globally_proven
