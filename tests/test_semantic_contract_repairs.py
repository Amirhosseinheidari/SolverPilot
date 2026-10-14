"""Independent semantic oracles for the October audit reproductions."""
from dataclasses import replace
from fractions import Fraction
import pickle
from types import SimpleNamespace
import numpy as np
import pytest
from solverpilot.model import Model


@pytest.mark.parametrize('power', [2.5, -.5, np.float64(2), True, '2', np.nan, np.inf])
def test_power_never_silently_changes_mathematics(power):
    m=Model(); x=m.variable()
    with pytest.raises(ValueError): x**power


def test_integer_power_is_preserved():
    m=Model(); x=m.variable(); m.minimize(x**np.int64(3))
    from solverpilot.nlp.ad import evaluate_node
    p=m.compile().execution_ir
    assert evaluate_node(p,p.objective_node,[2]) == 8


@pytest.mark.parametrize('discrete', [False, True])
def test_original_validation_requires_the_same_snapshot(discrete):
    m=Model(); x=m.variable(lower=0,upper=3); p=m.parameter(value=1.)
    m.add(x>=p)
    if discrete:
        z=m.binary(); m.minimize(x**2+z); point=[1.,0.]
    else:
        m.minimize(x.sin()); point=[1.]
    compiled=m.compile(); assert compiled.validate_original(m,point).valid
    p.value=2.
    with pytest.raises(ValueError,match='snapshot'): compiled.validate_original(m,point)
    assert not compiled.refresh(m).validate_original(m,point).valid


@pytest.mark.parametrize('bad', [[2.], [2.,np.nan], None, [2.,np.inf]])
def test_minlp_invalid_shape_and_nonfinite_discrete_are_reports(bad):
    from solverpilot.minlp import validate_minlp_solution
    m=Model(); x=m.variable(lower=0,upper=3); z=m.binary(); m.minimize((x-2)**2+z)
    p=m.compile().execution_ir; r=validate_minlp_solution(p,bad)
    assert not r.valid and r.max_integrality_violation == np.inf
    assert r.binary_violations == (1,)


def test_binary_enumeration_respects_fixed_original_domain():
    from solverpilot.minlp import solve_binary_enumeration
    from solverpilot.nlp import validate_nlp_solution
    class Oracle:
        def __init__(self): self.bounds=[]
        def solve(self,p,**kwargs):
            self.bounds.append(p.variable_lower[1]); x=np.array([0.,p.variable_lower[1]])
            return SimpleNamespace(x=x,validation=validate_nlp_solution(p,x),local_optimal_candidate=True,backend_status='analytic')
    m=Model(); x=m.variable(lower=0,upper=3); z=m.variable(lower=1,upper=1,domain='binary'); m.minimize(x**2+z)
    backend=Oracle(); r=solve_binary_enumeration(m.compile().execution_ir,nlp_backend=backend,max_assignments=1)
    assert backend.bounds == [1.]
    assert r.validation_valid and r.globally_proven and r.objective == 1.


def test_result_storage_and_trace_parameters_have_owned_metadata():
    from solverpilot._immutability import readonly_array
    from solverpilot.trace.schema import SolveTrace
    a=readonly_array([1.])
    for name,value in [('shape',(1,1)),('dtype',np.uint64)]:
        with pytest.raises(ValueError): setattr(a,name,value)
    b=pickle.loads(pickle.dumps(a)); assert np.array_equal(a,b)
    with pytest.raises(ValueError): b.dtype=np.uint64
    c=a.copy(); c[0]=2; assert a[0]==1
    supplied={'validation_tolerances':{'feasibility':1e-7}}
    trace=SolveTrace('s','d','LP',parameters=supplied); supplied['validation_tolerances']['feasibility']=42
    assert trace.parameters['validation_tolerances']['feasibility']==1e-7
    with pytest.raises(TypeError): trace.parameters['validation_tolerances']['feasibility']=42
    assert pickle.loads(pickle.dumps(trace)).to_dict()==trace.to_dict()


@pytest.mark.parametrize('candidate', [.9, -0.1, np.nan, np.inf, '0'])
def test_cp_validation_never_truncates_candidates(candidate):
    from solverpilot.cp import CPModel,validate_cp_solution
    m=CPModel(); x=m.int_var(0,1); m.add(x==0); m.minimize(x)
    r=validate_cp_solution(m.compile(),{x.var_id:candidate},objective_reported=candidate)
    assert not r.valid


def test_cp_constructors_reject_fractional_domains_and_objectives():
    from solverpilot.cp import CPModel,validate_cp_solution
    from solverpilot.cp.ir import CPLinearExprIR
    m=CPModel()
    with pytest.raises(ValueError): m.int_var(0,.9)
    with pytest.raises(ValueError): m.int_var_from_values([0,.9])
    with pytest.raises(ValueError): CPLinearExprIR(((0,.5),),0)
    x=m.int_var(0,1); m.minimize(x)
    assert not validate_cp_solution(m.compile(),{0:0},objective_reported=.9).valid
    assert validate_cp_solution(m.compile(),{0:0.0},objective_reported=0).valid


@pytest.mark.parametrize('method', ['solve_tsp_brute_force','solve_tsp_held_karp'])
def test_exact_tsp_compares_costs_before_float_rounding(method):
    from solverpilot.applications import tsp
    instance=tsp.TSPInstance.from_distance_matrix([[0,1e16,1e16],[0,0,1],[0,.5,0]])
    result=getattr(tsp,method)(instance)
    exact=sum((Fraction(instance.distances.values[a][b]) for a,b in zip(result.route,result.route[1:])),Fraction())
    assert exact == Fraction(10**16)+Fraction(1,2)
    assert result.optimality_proven


@pytest.mark.parametrize('kind', ['power','generalized_power'])
def test_power_cone_boundary_is_monotone_in_tolerance(kind):
    from solverpilot.conic.validation import _cone_check
    from solverpilot.conic.ir import ConeKind
    k=ConeKind(kind); kwargs={'alpha':.5} if kind=='power' else {'weights':(.5,.5)}
    for atol in (0.,1e-7,1.):
        r=_cone_check(k,np.array([1e300]*3),atol=atol,rtol=0.,source_id=None,**kwargs)
        assert r.valid and r.violation==0.


@pytest.mark.parametrize('case', range(13))
def test_casadi_values_and_derivatives_match_independent_numpy(case):
    pytest.importorskip('casadi')
    from solverpilot.nlp import compile_nlp_model,NLPDerivativeEngine
    builders=[
        (lambda x:x[1], lambda x:x[1]),
        (lambda x:x[:1], lambda x:x[:1]),
        (lambda x:x[:,1], lambda x:x[:,1]),
        (lambda x:x[::-1,::-1], lambda x:x[::-1,::-1]),
        (lambda x:x+np.array([1.,2.,3.]), lambda x:x+np.array([1.,2.,3.])),
        (lambda x:x+np.array([[1.],[2.]]), lambda x:x+np.array([[1.],[2.]])),
        (lambda x:np.sum(x,0), lambda x:np.sum(x,0)),
        (lambda x:np.sum(x,axis=1,keepdims=True), lambda x:np.sum(x,axis=1,keepdims=True)),
        (lambda x:x@np.array([1.,2.,3.]), lambda x:x@np.array([1.,2.,3.])),
        (lambda x:np.array([1.,2.])@x, lambda x:np.array([1.,2.])@x),
        (lambda x:x[0]@np.array([1.,2.,3.]), lambda x:x[0]@np.array([1.,2.,3.])),
        (lambda x:x@np.array([[1.,2.],[3.,4.],[5.,6.]]), lambda x:x@np.array([[1.,2.],[3.,4.],[5.,6.]])),
        (lambda x:x[:0], lambda x:x[:0]),
    ]
    symbolic,numeric=builders[case]; m=Model(); x=m.variable((2,3),lower=-10,upper=10)
    e=symbolic(x); m.minimize(e.sin().sum()); m.add(e<=100)
    p=compile_nlp_model(m).execution_ir; ad=NLPDerivativeEngine(p)
    point=np.random.default_rng(81+case).uniform(-1,1,6)
    oracle=lambda z: float(np.sin(numeric(z.reshape(2,3))).sum())
    assert ad.objective(point)==pytest.approx(oracle(point),abs=1e-12)
    assert np.allclose(ad.constraints(point),numeric(point.reshape(2,3)).reshape(-1)-100,atol=1e-12)
    grad=[]
    for i in range(6):
        step=np.eye(6)[i]*1e-6; grad.append((oracle(point+step)-oracle(point-step))/(2e-6))
    assert np.allclose(ad.gradient(point),grad,atol=2e-7,rtol=2e-7)


def test_local_nlp_maximization_original_objective_and_stationarity():
    pytest.importorskip('casadi')
    from solverpilot.nlp import CasadiIpoptBackend
    m=Model(); x=m.variable(lower=0,upper=1); m.maximize(x.exp())
    r=m.compile().solve(backend=CasadiIpoptBackend())
    assert r.validation.valid and r.local_optimal_candidate and not r.globally_proven
    assert r.x[0]==pytest.approx(1,abs=1e-6)
    assert r.objective_reported==pytest.approx(np.e,rel=1e-6)


def test_vector_oa_curvature_and_each_row_cut():
    pytest.importorskip('casadi')
    from solverpilot.minlp.orchestrator import _oa_cuts_at
    m=Model(); x=m.variable(2,lower=-2,upper=2); z=m.binary(); m.minimize(((x-z)**2).sum()+z)
    m.add((x-z)**2<=1)
    p=m.compile().execution_ir; cuts=_oa_cuts_at(p,np.array([.5,.25,0.]))
    assert p.convexity_certificate.objective_curvature=='convex' and len(cuts)==3
    for row,lo,hi in cuts[:2]: assert np.isfinite(hi) and lo==-np.inf


def test_portfolio_accounts_for_last_call_overrun(monkeypatch):
    from solverpilot.runtime import portfolio
    from solverpilot import LinearProblem,SolveBudget
    from solverpilot.backends import ScipyHighsLPBackend
    from solverpilot.validate import PublicStatus
    clock=[0.]
    monkeypatch.setattr(portfolio,'perf_counter',lambda:clock[0])
    def execute(*args,**kwargs):
        clock[0]=2.
        return SimpleNamespace(status=PublicStatus.VALID_OPTIMAL,objective=0.,validation=SimpleNamespace(valid=True))
    monkeypatch.setattr(portfolio,'execute',execute)
    p=LinearProblem.from_data(A=[[1.]],c=[1.],variable_lower=[0.],variable_upper=[1.],constraint_lower=[0.],constraint_upper=[1.])
    result=portfolio.execute_portfolio(p,[ScipyHighsLPBackend()],budget=SolveBudget(wall_time_s=1.))
    assert result.chosen is not None and result.budget_exhausted and result.stopped_reason=='valid_optimal_found'
