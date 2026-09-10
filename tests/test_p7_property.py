import pytest
pytest.importorskip("casadi", reason="optional CasADi extra not installed in core CI")

import numpy as np
from solverpilot.model import Model, sin, cos, exp, tanh
from solverpilot.nlp import NLPDerivativeEngine, compile_nlp_model


def test_random_smooth_derivatives_300_cases():
    rng=np.random.default_rng(20260906)
    for _ in range(300):
        m=Model(); x=m.variable(3,lower=-2,upper=2); a=rng.normal(size=3); b=rng.normal(size=3)
        expr=((a*x).sum())**2 + 0.2*sin(x[0]*x[1]) + 0.1*cos(x[2]) + 0.05*exp(0.2*x[0]) + 0.03*tanh(x[1])
        m.minimize(expr); m.add((b*x).sum()+0.2*sin(x[0]) <= 2.5)
        ad=NLPDerivativeEngine(compile_nlp_model(m).execution_ir); z=rng.uniform(-0.8,0.8,size=3); h=1e-6
        gfd=np.zeros(3); Jfd=np.zeros((1,3))
        for i in range(3):
            zp=z.copy(); zm=z.copy(); zp[i]+=h; zm[i]-=h
            gfd[i]=(ad.objective(zp)-ad.objective(zm))/(2*h)
            Jfd[:,i]=(ad.constraints(zp)-ad.constraints(zm))/(2*h)
        assert np.allclose(ad.gradient(z),gfd,atol=3e-6,rtol=3e-6)
        assert np.allclose(ad.jacobian(z),Jfd,atol=3e-6,rtol=3e-6)


def test_random_convex_like_nlp_solves_80_cases():
    rng=np.random.default_rng(17)
    for _ in range(80):
        center=rng.uniform(-1,1,size=2); m=Model(); x=m.variable(2,lower=-3,upper=3)
        m.minimize((x[0]-center[0])**4+(x[1]-center[1])**4+0.1*((x[0]-center[0])**2+(x[1]-center[1])**2))
        r=m.solve(x0=[0,0]); assert r.validation.valid; assert r.local_optimal_candidate; assert np.allclose(r.x,center,atol=3e-4); assert not r.globally_proven
