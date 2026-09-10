import numpy as np
import pytest

pytest.importorskip("casadi", reason="optional CasADi extra not installed in core CI")

from solverpilot.model import Model, sin, cos, exp, log, tanh
from solverpilot.nlp import NLPDerivativeEngine, NLPProblem, compile_nlp_model, validate_nlp_solution


def finite_grad(fun,x,h=1e-6):
    x=np.asarray(x,float); g=np.zeros_like(x)
    for i in range(x.size):
        xp=x.copy(); xm=x.copy(); xp[i]+=h; xm[i]-=h
        g[i]=(fun(xp)-fun(xm))/(2*h)
    return g

def finite_jac(fun,x,m,h=1e-6):
    x=np.asarray(x,float); J=np.zeros((m,x.size))
    for i in range(x.size):
        xp=x.copy(); xm=x.copy(); xp[i]+=h; xm[i]-=h
        J[:,i]=(fun(xp)-fun(xm))/(2*h)
    return J


def build_model():
    m=Model('nlp'); x=m.variable(3,lower=[-2,-2,0.2],upper=[2,2,3])
    p=m.parameter(value=0.3,name='p')
    m.minimize((x[0]-0.7)**2 + exp(x[1])*0.1 + sin(x[0]*x[1])*p + log(x[2]))
    m.add(sin(x[0])+x[1]**3 <= 1.2)
    m.add(x[0]*x[2] + tanh(x[1]) >= -0.5)
    return m,x,p


def test_compile_routes_nonlinear_model_to_nlp_ir():
    m,_,_=build_model(); c=m.compile(); assert isinstance(c.execution_ir,NLPProblem); assert c.schema_version=='solverpilot.compiled-model.p7.v1'


def test_semantic_hash_captures_power_exponent():
    a=Model(); xa=a.variable(); a.minimize(xa**3)
    b=Model(); xb=b.variable(); b.minimize(xb**4)
    # namespaces differ too, but serial payload must explicitly include exponent
    pa=a._serialize_node(a.objective.expression._node,include_parameter_values=False)
    pb=b._serialize_node(b.objective.expression._node,include_parameter_values=False)
    assert pa['exponent']==3 and pb['exponent']==4


def test_ad_gradient_jacobian_match_finite_difference():
    m,_,_=build_model(); prob=compile_nlp_model(m).execution_ir; ad=NLPDerivativeEngine(prob); x=np.array([0.2,-0.3,1.4])
    f=lambda z: ad.objective(z); gg=lambda z: ad.constraints(z)
    assert np.allclose(ad.gradient(x),finite_grad(f,x),atol=2e-6,rtol=2e-6)
    assert np.allclose(ad.jacobian(x),finite_jac(gg,x,prob.n_constraints),atol=2e-6,rtol=2e-6)


def test_hessian_matches_gradient_difference():
    m,_,_=build_model(); prob=compile_nlp_model(m).execution_ir; ad=NLPDerivativeEngine(prob); x=np.array([0.2,-0.3,1.4]); lam=np.zeros(prob.n_constraints)
    H=ad.lagrangian_hessian(x,lam); h=2e-5; Hfd=np.zeros_like(H)
    for i in range(x.size):
        xp=x.copy(); xm=x.copy(); xp[i]+=h; xm[i]-=h
        Hfd[:,i]=(ad.gradient(xp)-ad.gradient(xm))/(2*h)
    assert np.allclose(H,Hfd,atol=2e-5,rtol=2e-5)
    assert np.allclose(H,H.T,atol=1e-12)


def test_jvp_vjp_match_dense_jacobian():
    m,_,_=build_model(); prob=compile_nlp_model(m).execution_ir; ad=NLPDerivativeEngine(prob); x=np.array([0.2,-0.3,1.4]); J=ad.jacobian(x)
    v=np.array([.4,-.2,.7]); w=np.array([.6,-.3])
    assert np.allclose(ad.jvp(x,v),J@v,atol=1e-10)
    assert np.allclose(ad.vjp(x,w),J.T@w,atol=1e-10)


def test_sparsity_is_structural_and_nonempty():
    m,_,_=build_model(); ad=NLPDerivativeEngine(compile_nlp_model(m).execution_ir)
    assert len(ad.jacobian_sparsity())>0; assert len(ad.hessian_sparsity())>0


def test_parameter_change_keeps_semantic_hash_and_changes_data_hash():
    m,_,p=build_model(); c1=compile_nlp_model(m); sh=m.semantic_hash; dh=m.data_hash; p.value=0.9; c2=compile_nlp_model(m)
    assert m.semantic_hash==sh; assert m.data_hash!=dh; assert c1.semantic_hash==c2.semantic_hash; assert c1.data_hash!=c2.data_hash


def test_validation_recomputes_original_nonlinear_objective_and_constraints():
    m,_,_=build_model(); prob=compile_nlp_model(m).execution_ir; x=np.array([0.1,-0.1,1.2]); ad=NLPDerivativeEngine(prob); obj=ad.objective(x)
    r=validate_nlp_solution(prob,x,objective_reported=obj); assert r.valid; assert r.objective_difference==pytest.approx(0,abs=1e-12)


def test_validation_rejects_domain_nan():
    m=Model(); x=m.variable(lower=-2,upper=2); m.minimize(log(x)); prob=compile_nlp_model(m).execution_ir
    r=validate_nlp_solution(prob,np.array([-1.])); assert not r.valid; assert not r.finite


def test_nlp_rejects_integer_variables():
    m=Model(); x=m.integer(lower=0,upper=2); m.minimize(sin(x))
    with pytest.raises(Exception,match='continuous variables'): compile_nlp_model(m)

def test_symbolic_division_routes_to_nlp_and_derivatives_match():
    m=Model(); x=m.variable(2,lower=[0.2,-1],upper=[3,1]); m.minimize(x[0]/(1+x[1]**2) + sin(x[1]))
    c=m.compile(); assert isinstance(c.execution_ir,NLPProblem)
    ad=NLPDerivativeEngine(c.execution_ir); z=np.array([1.2,0.3]); assert np.allclose(ad.gradient(z),finite_grad(ad.objective,z),atol=2e-6,rtol=2e-6)


def test_symbolic_division_singularity_is_not_accepted_as_valid():
    m=Model(); x=m.variable(lower=-1,upper=1); m.minimize(1/x); prob=compile_nlp_model(m).execution_ir
    r=validate_nlp_solution(prob,np.array([0.])); assert not r.valid; assert not r.finite
