import importlib.util
import numpy as np
import pytest

CASADI_AVAILABLE = importlib.util.find_spec("casadi") is not None
requires_casadi = pytest.mark.skipif(not CASADI_AVAILABLE, reason="optional CasADi MINLP solve path not installed in core CI")
import solverpilot as om
from solverpilot.model.errors import CompileError
from solverpilot.minlp import MINLPProblem, solve_binary_enumeration, solve_outer_approximation, validate_minlp_solution


def make_square_model():
    m=om.Model('square-minlp')
    x=m.variable(lower=0,upper=3,name='x')
    z=m.binary(name='z')
    m.minimize((x-2)**2 + 0.2*(1-z))
    return m,x,z


def test_compile_routes_binary_quadratic_to_minlp():
    m,_,_=make_square_model(); c=m.compile()
    assert isinstance(c.execution_ir,MINLPProblem)
    assert c.schema_version=='solverpilot.compiled-model.p8.v1'
    assert c.execution_ir.convexity_certificate.objective_curvature=='convex'


@requires_casadi
def test_binary_enumeration_proves_global_optimum():
    m,_,_=make_square_model(); p=m.compile().execution_ir
    r=solve_binary_enumeration(p)
    assert r.globally_proven
    assert r.status=='proven_optimal'
    assert np.allclose(r.x,[2,1],atol=1e-5)
    assert r.objective == pytest.approx(0,abs=1e-6)


@requires_casadi
def test_outer_approximation_proves_same_optimum():
    m,_,_=make_square_model(); p=m.compile().execution_ir
    e=solve_binary_enumeration(p); r=solve_outer_approximation(p)
    assert r.globally_proven
    assert r.objective == pytest.approx(e.objective,abs=2e-6)
    assert r.lower_bound <= r.objective + 1e-6


@requires_casadi
def test_compiled_solve_defaults_to_oa():
    m,_,_=make_square_model(); r=m.compile().solve()
    assert r.algorithm=='outer_approximation' and r.globally_proven


def test_integrality_validator_rejects_fractional_binary():
    m,_,_=make_square_model(); p=m.compile().execution_ir
    rep=validate_minlp_solution(p,np.array([2.0,0.5]))
    assert not rep.valid and rep.max_integrality_violation==pytest.approx(0.5)


def test_rejects_bilinear_nonconvex_objective():
    m=om.Model(); x=m.variable(lower=0,upper=2); z=m.binary(); m.minimize(x*z)
    with pytest.raises(CompileError,match='convex objective'):
        m.compile()


def test_rejects_general_integer_v1():
    m=om.Model(); x=m.variable(lower=0,upper=3); k=m.integer(lower=0,upper=2); m.minimize((x-k)**2)
    with pytest.raises(CompileError,match='binary discrete'):
        m.compile()


def test_rejects_nonlinear_equality():
    m=om.Model(); x=m.variable(lower=0,upper=2); z=m.binary(); m.minimize((x-1)**2 + z)
    m.add((x-z)**2 == 0)
    with pytest.raises(CompileError,match='nonlinear equality'):
        m.compile()


@requires_casadi
def test_convex_nonlinear_upper_constraint_supported():
    m=om.Model(); x=m.variable(lower=0,upper=3); z=m.binary(); m.minimize((x-2)**2 + 0.1*(1-z)); m.add((x-z)**2 <= 1.0)
    c=m.compile(); assert isinstance(c.execution_ir,MINLPProblem)
    r=solve_binary_enumeration(c.execution_ir); assert r.globally_proven
    assert c.validate_original(m,r.x).valid


@requires_casadi
def test_exp_affine_objective_certificate():
    m=om.Model(); x=m.variable(lower=-1,upper=1); z=m.binary(); m.minimize(om.exp(x-z) + 0.1*z)
    c=m.compile(); assert c.execution_ir.convexity_certificate.objective_curvature=='convex'
    r=solve_binary_enumeration(c.execution_ir); assert r.globally_proven


def test_parameter_change_recertifies_data_hash_and_compilation_hash():
    m=om.Model(); x=m.variable(lower=0,upper=3); z=m.binary(); p=m.parameter(value=1.0); m.minimize(p*(x-2)**2 + 0.1*(1-z))
    c1=m.compile(); p.value=2.0; c2=m.compile()
    assert c1.semantic_hash==c2.semantic_hash
    assert c1.data_hash!=c2.data_hash
    assert c1.compilation_hash!=c2.compilation_hash


def test_bad_parameter_sign_can_make_convexity_uncertified():
    m=om.Model(); x=m.variable(lower=0,upper=3); z=m.binary(); p=m.parameter(value=1.0); m.minimize(p*(x-2)**2 + z)
    m.compile(); p.value=-1.0
    with pytest.raises(CompileError,match='convex objective'):
        m.compile()


def test_affine_minlp_does_not_route_to_p8():
    m=om.Model(); x=m.variable(lower=0,upper=2); z=m.binary(); m.minimize(x+z); m.add(x+z>=1)
    c=m.compile(); assert not isinstance(c.execution_ir,MINLPProblem)


def test_unresolved_nlp_backend_prevents_global_proof():
    m,_,_=make_square_model(); p=m.compile().execution_ir
    class Bad:
        def solve(self,*a,**k):
            from solverpilot.nlp.backend import NLPSolveResult
            from solverpilot.nlp.validation import validate_nlp_solution
            sub=a[0]
            rep=validate_nlp_solution(sub,None)
            return NLPSolveResult('bad','failed',None,None,rep,False,False,None,None,None,{})
    r=solve_binary_enumeration(p,nlp_backend=Bad())
    assert not r.globally_proven and r.status=='unknown'


@requires_casadi
def test_oa_trace_contains_bounds_and_assignment():
    m=om.Model(); x=m.variable(lower=-2,upper=2); z=m.binary(); m.minimize((x-0.2)**2 + (z-0.4)**2)
    r=solve_outer_approximation(m.compile().execution_ir)
    assert r.globally_proven and r.iterations
    assert all(len(i.binary_assignment)==1 for i in r.iterations)
    assert r.lower_bound is not None

@requires_casadi
def test_p8_conformance_report_passes():
    rep=om.conform_minlp_orchestrator()
    assert rep.passed
    assert len(rep.checks)>=4


def test_capability_requirements_classify_minlp():
    m,_,_=make_square_model(); p=m.compile().execution_ir
    req=om.requirements_v2_for(p)
    assert om.CapabilityKey.PROBLEM_MINLP in req.required
    assert om.CapabilityKey.CONSTRAINT_NONLINEAR in req.required


def test_semantic_schema_marks_discrete_nonlinear_as_p8():
    m,_,_=make_square_model()
    assert m._semantic_payload(include_parameter_values=False)['schema']=='solverpilot.semantic-model.p8.v1'


@requires_casadi
def test_fractional_oa_requires_real_iterations():
    m=om.Model(); x=m.variable(lower=-2,upper=2); z=m.binary(); m.minimize((x-0.1)**2+(z-0.35)**2)
    r=solve_outer_approximation(m.compile().execution_ir)
    assert r.globally_proven and len(r.iterations)>=1

def test_bound_closure_detects_large_inversion():
    from solverpilot.minlp.orchestrator import _bound_closure
    closed,gap,inv,tol=_bound_closure(10.0,10.2,1e-6)
    assert not closed and gap==0 and inv==pytest.approx(0.2)
    closed2,_,inv2,_=_bound_closure(100000.0,100000.005,1e-7)
    assert closed2 and inv2==pytest.approx(0.005)
