from dataclasses import asdict,replace,dataclass
from decimal import Decimal,localcontext
from fractions import Fraction
from types import SimpleNamespace
import numpy as np
import pytest
from scipy import sparse
from solverpilot import LinearProblem,QuadraticProblem,SolveBudget
from solverpilot.model import Model


def lp():
    return LinearProblem.from_data(A=[[1.]],c=[1.],variable_lower=[0.],variable_upper=[2.],constraint_lower=[1.],constraint_upper=[np.inf])


def test_workspace_reuses_only_preparation_and_invalidates_data():
    from solverpilot.validate.workspace import VerificationWorkspace
    from solverpilot.validate.optimality import verify_optimality
    p=lp(); w=VerificationWorkspace(max_entries=2,max_coefficient_terms=10)
    y=np.array([-1.,0.]); first=verify_optimality(p,[1.],y,workspace=w)
    assert first.verified
    assert asdict(verify_optimality(p,[1.],y,workspace=w))==asdict(first)
    assert w.info()['hits']>0
    assert not verify_optimality(p,[0.],y,workspace=w).verified
    changed=replace(p,A=sparse.csr_matrix([[2.]]),constraint_lower=np.array([2.]))
    assert verify_optimality(changed,[1.],[-.5,0.],workspace=w).verified
    assert w.info()['entries']<=2 and w.info()['coefficient_terms']<=10
    small=VerificationWorkspace(max_coefficient_terms=0)
    assert verify_optimality(p,[1.],y,workspace=small).verified and small.info()['entries']==0
    w.clear(); assert w.info()['entries']==0


def test_proof_budget_is_fail_closed_and_required_retains_result():
    from solverpilot.runtime import solve
    from solverpilot.validate.workspace import ProofPolicy,ProofRequirementNotMetError
    from solverpilot.validate.optimality import verify_optimality
    check=verify_optimality(lp(),[1.],[-1.,0.],proof_policy=ProofPolicy('budgeted',max_terms=1))
    assert not check.verified and 'limit' in check.reason
    result=solve(lp(),backend='scipy-highs-ds',proof_policy=ProofPolicy('required'))
    assert result.optimality_evidence.independently_verified_optimal
    with pytest.raises(ProofRequirementNotMetError) as failure:
        solve(lp(),backend='scipy-highs-ds',proof_policy=ProofPolicy('required',max_terms=1))
    assert failure.value.result.validation.valid and not failure.value.result.optimality_evidence.independently_verified_optimal


def test_lexicographic_aggregate_clock_preserves_completed_stages(monkeypatch):
    import solverpilot.runtime.budgeting as budgeting
    import solverpilot.runtime.multiobjective as multi
    from solverpilot.validate import PublicStatus
    clock=[0.]; calls=[]; monkeypatch.setattr(budgeting,'perf_counter',lambda:clock[0])
    def solve(p,**kwargs):
        calls.append(kwargs['budget'].wall_time_s); clock[0]+=.6
        return SimpleNamespace(status=PublicStatus.VALID_OPTIMAL,validation=SimpleNamespace(valid=True),x=np.array([1.]))
    monkeypatch.setattr(multi,'solve',solve)
    r=multi.solve_lexicographic(lp(),[[1.],[1.],[1.]],budget=SolveBudget(wall_time_s=1.))
    assert calls==pytest.approx([1.,.4]) and not r.completed
    assert r.x.tolist()==[1.] and r.budget_report['budget_exhausted']


def test_minlp_aggregate_exhaustion_keeps_original_valid_incumbent(monkeypatch):
    from solverpilot.minlp import solve_binary_enumeration
    from solverpilot.nlp import validate_nlp_solution
    import solverpilot.runtime.budgeting as budgeting
    clock=[0.]; calls=[]; monkeypatch.setattr(budgeting,'perf_counter',lambda:clock[0])
    @dataclass
    class Oracle:
        time_limit_s:float|None=None
        def solve(self,p,**kwargs):
            calls.append(self.time_limit_s); clock[0]+=1.1
            x=np.array([0.,p.variable_lower[1]])
            return SimpleNamespace(x=x,validation=validate_nlp_solution(p,x),local_optimal_candidate=True,backend_status='analytic')
    m=Model(); x=m.variable(lower=0,upper=2); z=m.binary(); m.minimize(x**2+z)
    r=solve_binary_enumeration(m.compile().execution_ir,nlp_backend=Oracle(),budget=SolveBudget(wall_time_s=1.))
    assert len(calls)==1 and r.validation_valid and r.objective==0.
    assert not r.globally_proven and r.raw_statistics['call_budget']['budget_exhausted']
    assert len(r.iterations)==1 and r.raw_statistics['completed_iterations']==1


def test_sparse_psd_components_and_negative_perturbations():
    from solverpilot.problem.curvature import certify_psd
    block=np.array([[1.,2.],[2.,4.]])
    P=sparse.block_diag([sparse.csr_matrix(block)]*40,format='csr')
    check=certify_psd(P); assert check.certified and 'component' in check.method
    assert not certify_psd(P,max_component_work=1).certified
    P[0,0]=np.nextafter(1.,0.)
    assert certify_psd(P).status=='not_psd'


@pytest.mark.parametrize('kind,point',[('exp',-1.),('exp',2.),('log',.2),('log',2.),('sqrt',2.),('sqrt',0.)])
def test_rational_elementary_enclosures(kind,point):
    from solverpilot.globalopt.intervals import _exp_point,_log_point,_sqrt_point
    value={'exp':_exp_point,'log':_log_point,'sqrt':_sqrt_point}[kind](Fraction(point))
    if kind=='sqrt':
        assert value.lo**2<=Fraction(point)<=value.hi**2
    else:
        with localcontext() as ctx:
            ctx.prec=100; x=Decimal.from_float(point)
            oracle=x.exp() if kind=='exp' else x.ln()
            assert Decimal(value.lo.numerator)/Decimal(value.lo.denominator)<=oracle<=Decimal(value.hi.numerator)/Decimal(value.hi.denominator)


def test_global_composed_domains_and_clone():
    from solverpilot.globalopt import absolute,maximum
    m=Model(); x=m.variable(lower=-1,upper=1); m.minimize(x.exp().log()+absolute(x))
    p=m.compile(target='global'); clone=m.clone(); clone.compile(target='global')
    assert p.execution_ir.relaxation.objective_sense=='minimize'
    m2=Model(); y=m2.variable(2,lower=-1,upper=1); m2.minimize(maximum(y)); m2.clone().compile(target='global')


@pytest.mark.parametrize('alpha,expected',[(0.,1.),(.5,1.6),(.75,2.2)])
def test_cvar_matches_independent_discrete_tail_formula(alpha,expected):
    from solverpilot.model.robust import minimize_cvar
    m=Model(); x=m.variable(1,lower=1,upper=1)
    # losses [0,1,3], probabilities [.5,.3,.2]
    helper=minimize_cvar(m,x,[[0.],[1.],[3.]],probabilities=[.5,.3,.2],confidence=alpha)
    r=m.compile().solve(backend='scipy-highs-ds')
    losses=np.array([0.,1.,3.]); probabilities=np.array([.5,.3,.2])
    oracle=min(eta+probabilities@np.maximum(losses-eta,0)/(1-alpha) for eta in losses)
    assert r.validation.valid and r.objective==pytest.approx(oracle)
    with pytest.raises(ValueError): helper.probabilities.dtype=np.uint64


@pytest.mark.parametrize('kwargs',[{'confidence':1.},{'confidence':True},{'probabilities':[.1,.1]},{'probabilities':[-1.,2.]}])
def test_cvar_invalid_contract(kwargs):
    from solverpilot.model.robust import minimize_cvar
    m=Model(); x=m.variable(1)
    with pytest.raises(ValueError): minimize_cvar(m,x,[[1.],[2.]],**kwargs)


def test_ellipsoid_analytic_support():
    pytest.importorskip('clarabel')
    from solverpilot.model.robust import ellipsoidal_leq
    from solverpilot.conic import ClarabelBackend
    m=Model(); x=m.variable(1,lower=0); ellipsoidal_leq(m,[1.],[[2.]],x,3.,radius=.5)
    m.minimize(-x.sum()); r=m.compile().solve(backend=ClarabelBackend(execution_profile='chordal'))
    assert r.validation.valid and r.x[0]==pytest.approx(1.5,abs=1e-6)
    assert r.raw_statistics['chordal_decomposition_enabled']


def test_qp_sensitivity_matches_analytic_and_finite_difference():
    pytest.importorskip('osqp')
    from solverpilot.backends import OSQPNativeBackend
    from solverpilot.runtime import solve
    from solverpilot.sensitivity import differentiate_qp
    p=QuadraticProblem.from_data(P=[[2.]],q=[-4.],A=sparse.csr_matrix((0,1)),constraint_lower=[],constraint_upper=[],variable_lower=[-10.],variable_upper=[10.])
    backend=OSQPNativeBackend(eps_abs=1e-10,eps_rel=1e-10)
    r=solve(p,backend=backend); sensitivity=differentiate_qp(p,r,dc=[1.])
    assert sensitivity.status=='defined' and sensitivity.dx[0]==pytest.approx(-.5)
    h=1e-4
    points=[solve(replace(p,linear=replace(p.linear,c=np.array([-4.+sign*h]))),backend=backend).x[0] for sign in (-1,1)]
    assert sensitivity.dx[0]==pytest.approx((points[1]-points[0])/(2*h),abs=1e-5)
    changed=replace(p,linear=replace(p.linear,c=np.array([-5.])))
    with pytest.raises(ValueError): differentiate_qp(changed,r,dc=[1.])


def test_qp_sensitivity_weak_activity_is_undefined():
    pytest.importorskip('osqp')
    from solverpilot.runtime import solve
    from solverpilot.backends import OSQPNativeBackend
    from solverpilot.sensitivity import differentiate_qp
    p=QuadraticProblem.from_data(P=[[2.]],q=[0.],A=sparse.csr_matrix((0,1)),constraint_lower=[],constraint_upper=[],variable_lower=[0.],variable_upper=[2.])
    r=solve(p,backend=OSQPNativeBackend(eps_abs=1e-10,eps_rel=1e-10))
    s=differentiate_qp(p,r,dc=[1.]); assert s.status=='undefined' and s.dx is None


@pytest.mark.parametrize('cap',[{'max_terms':1},{'time_limit_s':1e-15}])
def test_proof_limits_include_termination_witnesses_and_recovery(cap):
    from solverpilot.runtime import solve,execute
    from solverpilot.backends import ScipyHighsBackend,BackendSolveResult
    from solverpilot.validate.workspace import ProofPolicy,ProofRequirementNotMetError
    p=replace(lp(),c=np.array([0.]),variable_upper=np.array([1.]),constraint_lower=np.array([2.]))
    class Backend:
        manifest=ScipyHighsBackend().manifest
        def is_available(self): return True
        def solve(self,p):
            return BackendSolveResult('infeasible',None,raw_statistics={'termination_certificate':{'problem_data_hash':p.data_hash,'kind':'infeasible','dual':[-1.,1.]}})
    r=execute(p,Backend(),proof_policy=ProofPolicy('budgeted',**cap))
    assert r.termination_certificate is None and not r.raw_statistics['proof_requirement_met']
    with pytest.raises(ProofRequirementNotMetError): execute(p,Backend(),proof_policy=ProofPolicy('required',**cap))
    r=solve(p,backend='scipy-highs-ds',certificate_recovery=1.,proof_policy=ProofPolicy('budgeted',**cap))
    assert r.termination_certificate is None


def test_required_mode_allows_authorized_recovery_before_enforcing():
    from solverpilot.runtime import solve
    from solverpilot.validate.workspace import ProofPolicy
    p=replace(lp(),c=np.array([0.]),variable_upper=np.array([1.]),constraint_lower=np.array([2.]))
    r=solve(p,backend='scipy-highs-ds',certificate_recovery=1.,proof_policy=ProofPolicy('required'))
    assert r.termination_certificate.verified and r.raw_statistics['proof_requirement_met']


def test_workspace_duplicate_cancellation_never_poisons_other_matrix():
    from solverpilot.validate.workspace import VerificationWorkspace
    from solverpilot.validate.optimality import verify_optimality
    w=VerificationWorkspace(); duplicate=sparse.csr_matrix(([1e16,1.,-1e16,1.],[0,0,0,0],[0,4]),shape=(1,1))
    w.prepare(duplicate)
    p=LinearProblem.from_data(A=sparse.csr_matrix((0,1)),c=[-1.],variable_lower=[-2.],variable_upper=[1.],constraint_lower=[],constraint_upper=[])
    assert asdict(verify_optimality(p,[1.],[1.],workspace=w))==asdict(verify_optimality(p,[1.],[1.]))


@pytest.mark.parametrize('lower',[-130.,-200.,-300.])
def test_global_structural_positive_exp_independent_of_grid(lower):
    m=Model(); x=m.variable(lower=lower,upper=lower+1); m.minimize(x.exp().log()); m.compile(target='global')


def test_cvar_zero_confidence_stays_expected_loss_with_probability_roundoff():
    from solverpilot.model.robust import minimize_cvar
    m=Model(); x=m.variable(1,lower=0,upper=0)
    minimize_cvar(m,x,[[0.],[0.]],probabilities=[.5,.4999999999995],confidence=0.)
    r=m.compile().solve(backend='scipy-highs-ds'); assert r.validation.valid and r.objective==0.
    with pytest.raises(ValueError,match='close to zero'):
        minimize_cvar(m,x,[[0.],[0.]],probabilities=[.5,.4999999999995],confidence=1e-15)


def test_oa_skips_forbidden_near_integral_rounding():
    pytest.importorskip('casadi')
    from solverpilot.minlp import solve_outer_approximation
    from solverpilot.nlp import validate_nlp_solution
    m=Model(); z=m.variable(domain='binary',lower=0,upper=.9999995); m.minimize((z-1)**2)
    class Oracle:
        def solve(self,p,**kwargs):
            x=np.array([p.variable_upper[0]])
            return SimpleNamespace(x=x,validation=validate_nlp_solution(p,x),local_optimal_candidate=True,backend_status='analytic')
    r=solve_outer_approximation(m.compile().execution_ir,nlp_backend=Oracle())
    assert r.validation_valid and r.x[0]==0.


def test_materialized_relaxation_has_its_own_lp_reference(tmp_path):
    pytest.importorskip('highspy')
    from solverpilot.benchmark.relaxations import materialize_lp_relaxation
    from solverpilot import read_mps
    source=tmp_path/'binary.mps'; destination=tmp_path/'continuous.mps'
    source.write_text("NAME DEMO\nROWS\n N COST\n G ROW\nCOLUMNS\n MARK0 'MARKER' 'INTORG'\n X COST 1 ROW 1\n MARK1 'MARKER' 'INTEND'\nRHS\n RHS ROW 0.5\nBOUNDS\n BV BND X\nENDATA\n")
    report=materialize_lp_relaxation(source,destination)
    assert read_mps(source).has_integer_variables and not read_mps(destination).has_integer_variables
    assert report['reference_verified'] and report['reference']==pytest.approx(.5)
    assert report['data_hash']!=report['source_data_hash'] and not report['production_authorized']
    with pytest.raises(FileExistsError): materialize_lp_relaxation(source,destination)


def test_specialized_report_and_export_preserve_local_scope(tmp_path):
    from solverpilot.nlp import validate_nlp_solution
    from solverpilot.nlp.backend import NLPSolveResult
    from solverpilot.runtime.manifest import save_run
    from solverpilot.reporting import explain_any
    m=Model(); x=m.variable(lower=-1,upper=1); m.minimize(x.exp())
    p=m.compile().execution_ir; validation=validate_nlp_solution(p,[0.],objective_reported=1.)
    r=NLPSolveResult('analytic','local',[0.],1.,validation,True,False,None,None,0.,{},p.data_hash)
    report=explain_any(r); assert report['summary']['optimality']=='local_candidate'
    assert 'no global proof' in report['interpretation']
    path=save_run(tmp_path/'run.json',p,r); assert path.exists()


@pytest.mark.parametrize('case',['upper','coefficient','singular','dependent'])
def test_qp_sensitivity_regular_and_degenerate_analytic_cases(case):
    from solverpilot.sensitivity import differentiate_qp
    if case=='upper':
        p=QuadraticProblem.from_data(P=[[2.]],q=[-4.],A=sparse.csr_matrix((0,1)),constraint_lower=[],constraint_upper=[],variable_lower=[0.],variable_upper=[1.])
        x=[1.]; y=[2.]; directions={'dupper':[1.]}; expected=1.
    elif case=='coefficient':
        p=QuadraticProblem.from_data(P=[[2.]],q=[0.],A=[[1.]],constraint_lower=[1.],constraint_upper=[np.inf],variable_lower=[-10.],variable_upper=[10.])
        x=[1.]; y=[-2.,0.]; directions={'dA':[[1.]]}; expected=-1.
    elif case=='singular':
        p=QuadraticProblem.from_data(P=[[0.]],q=[0.],A=sparse.csr_matrix((0,1)),constraint_lower=[],constraint_upper=[],variable_lower=[-1.],variable_upper=[1.])
        x=[0.]; y=[0.]; directions={'dc':[1.]}; expected=None
    else:
        p=QuadraticProblem.from_data(P=[[2.]],q=[0.],A=[[1.]],constraint_lower=[0.],constraint_upper=[0.],variable_lower=[0.],variable_upper=[0.])
        x=[0.]; y=[0.,0.]; directions={'dc':[1.]}; expected=None
    r=SimpleNamespace(trace=SimpleNamespace(problem_data_hash=p.data_hash),x=np.array(x),raw_statistics={'canonical_dual':y},execution_id='analytic')
    s=differentiate_qp(p,r,**directions)
    if expected is None: assert s.status=='undefined' and s.dx is None
    else: assert s.status=='defined' and s.dx[0]==pytest.approx(expected)


def test_scenario_aggregate_budget_excludes_consumer_idle_and_preserves_threads(monkeypatch):
    import solverpilot.scenarios as scenarios
    import solverpilot.runtime.budgeting as budgeting
    from solverpilot.runtime.options import SolveOptions
    from solverpilot.runtime import solve
    r=solve(lp(),backend='scipy-highs-ds'); clock=[0.]; calls=[]
    monkeypatch.setattr(budgeting,'perf_counter',lambda:clock[0]); monkeypatch.setattr(scenarios,'perf_counter',lambda:clock[0])
    class Session:
        def __init__(self,*args,**kwargs): pass
        def __enter__(self): return self
        def __exit__(self,*args): pass
        def solve(self,**kwargs): calls.append(kwargs['options'].budget); clock[0]+=.4; return r
        def problem_snapshot(self): return lp()
    monkeypatch.setattr(scenarios,'ReoptimizationSession',Session)
    stream=scenarios.scenario_sweep(Model(),[{}, {}, {}, {}],aggregate_budget=SolveBudget(wall_time_s=1.),options=SolveOptions(budget=SolveBudget(threads=2)))
    first=next(stream); clock[0]+=100.; rest=list(stream)
    assert len(rest)==2 and [b.wall_time_s for b in calls]==pytest.approx([1.,.6,.2])
    assert all(b.threads==2 for b in calls) and rest[-1].aggregate_budget_report['budget_exhausted']


def test_oa_near_integer_with_forbidden_rounded_value_never_proves_wrong_optimum():
    pytest.importorskip('casadi')
    from solverpilot.minlp import solve_outer_approximation,validate_minlp_solution
    from solverpilot.nlp import validate_nlp_solution
    m=Model(); z=m.variable(domain='binary',lower=0,upper=.99999995); b=m.binary(); m.minimize((z-1)**2+b**2)
    p=m.compile().execution_ir
    assert not validate_minlp_solution(p,[.99999995,0.]).valid
    class Oracle:
        def solve(self,sub,**kwargs):
            x=np.clip(np.array([1.,0.]),sub.variable_lower,sub.variable_upper)
            return SimpleNamespace(x=x,validation=validate_nlp_solution(sub,x),local_optimal_candidate=True,backend_status='analytic')
    r=solve_outer_approximation(p,nlp_backend=Oracle())
    assert not r.globally_proven or r.objective==pytest.approx(1.,abs=1e-7)


@pytest.mark.parametrize('kwargs',[{'max_iter':0},{'atol':-1.},{'max_iter':True}])
def test_single_assignment_oa_validates_controls_before_shortcut(kwargs):
    from solverpilot.minlp import solve_outer_approximation
    m=Model(); z=m.variable(domain='binary',lower=0,upper=0); m.minimize((z-1)**2)
    with pytest.raises(ValueError): solve_outer_approximation(m.compile().execution_ir,**kwargs)


def test_requested_recovery_preserves_an_already_verified_witness():
    from solverpilot.runtime import execute,solve
    from solverpilot.backends import ScipyHighsBackend,BackendSolveResult
    from solverpilot.validate.workspace import ProofPolicy
    p=replace(lp(),c=np.array([0.]),variable_upper=np.array([1.]),constraint_lower=np.array([2.]))
    class Backend:
        manifest=ScipyHighsBackend().manifest
        def is_available(self): return True
        def solve(self,p): return BackendSolveResult('infeasible',None,raw_statistics={'termination_certificate':{'problem_data_hash':p.data_hash,'kind':'infeasible','dual':[-1.,1.]}})
    first=execute(p,Backend(),proof_policy=ProofPolicy('required'))
    work=first.raw_statistics['verification_resources']['terms']
    r=solve(p,backend=Backend(),certificate_recovery=1.,proof_policy=ProofPolicy('required',max_terms=work))
    assert r.termination_certificate.verified and r.raw_statistics['proof_requirement_met']


def test_exact_sparse_factor_psd_witness_and_rounded_mismatch():
    from solverpilot.problem.curvature import certify_psd_factor
    R=sparse.csr_matrix(np.eye(40)); P=sparse.diags(np.arange(1,41,dtype=float))
    assert certify_psd_factor(P,R,weights=np.arange(1,41,dtype=float)).certified
    assert not certify_psd_factor(P,R).certified
    assert not certify_psd_factor(P,R,weights=np.arange(1,41,dtype=float),max_product_terms=1).certified
    fractional=np.array([[.1,.2]])
    assert not certify_psd_factor(fractional.T@fractional,fractional).certified
    with pytest.raises(ValueError): certify_psd_factor(P,R,weights=-np.ones(40))


def test_factor_witness_can_close_a_large_connected_psd_check():
    from solverpilot.validate.optimality import verify_optimality
    R=np.vstack([np.ones((1,40)),np.eye(40)]); weights=np.r_[1.,np.full(40,.125)]; P=np.ones((40,40))+.125*np.eye(40)
    p=QuadraticProblem.from_data(P=P,q=np.zeros(40),A=sparse.csr_matrix((0,40)),constraint_lower=[],constraint_upper=[],variable_lower=-np.ones(40),variable_upper=np.ones(40))
    assert not verify_optimality(p,np.zeros(40),np.zeros(40)).verified
    assert verify_optimality(p,np.zeros(40),np.zeros(40),curvature_witness={'factor':R,'weights':weights}).verified


@pytest.mark.parametrize('case',['target','coefficient','rhs'])
def test_analytic_semantic_parameter_chain_matches_known_derivative(case):
    pytest.importorskip('osqp')
    from solverpilot.backends import OSQPNativeBackend
    from solverpilot.model.parametric import parameter_data_direction,differentiate_parameter_qp
    m=Model(); x=m.variable(2,lower=-10,upper=10)
    if case=='target':
        p=m.parameter(2,value=[.5,1.],name='p'); m.minimize(((x-p)**2).sum()); direction=[2.,-1.]; expected=[2.,-1.]
    elif case=='coefficient':
        p=m.parameter(2,value=[1.,0.],name='p'); m.add(p@x>=1.); m.minimize((x**2).sum()); direction=[0.,1.]; expected=[0.,1.]
    else:
        p=m.parameter(value=1.,name='p'); m.add(x[0]>=p); m.minimize((x**2).sum()); direction=1.; expected=[1.,0.]
    before=m.data_hash; compiled=m.compile(); r=compiled.solve(backend=OSQPNativeBackend(eps_abs=1e-10,eps_rel=1e-10))
    data=parameter_data_direction(m,p,direction); s=differentiate_parameter_qp(m,r,p,direction)
    assert m.data_hash==before and s.status=='defined' and np.allclose(s.dx,expected,atol=2e-6)
    assert data.problem_data_hash==compiled.execution_ir.data_hash
    with pytest.raises(ValueError): data.dc.dtype=np.uint64
    other=Model(); foreign=other.parameter(value=1.)
    with pytest.raises(ValueError): parameter_data_direction(m,foreign)
    with pytest.raises(ValueError,match='budget'): parameter_data_direction(m,p,direction,max_expression_elements=1)


@pytest.mark.parametrize('field',['weights','diagonal'])
def test_factor_witness_owns_sign_checked_vectors(monkeypatch,field):
    import solverpilot.problem.curvature as curvature
    supplied=np.array([1.]); charge=curvature._charge_verification
    def mutate(terms): supplied[0]=-1.; charge(terms)
    monkeypatch.setattr(curvature,'_charge_verification',mutate)
    factor=[[1.]] if field=='weights' else [[0.]]
    check=curvature.certify_psd_factor([[-1.]],factor,**{field:supplied})
    assert not check.certified


def test_factor_witness_rejects_rounded_duplicate_representations():
    from solverpilot.problem.curvature import certify_psd_factor
    duplicate=sparse.csr_matrix(([1e16,-1.,-1e16,.5],[0,0,0,0],[0,4]),shape=(1,1))
    assert not certify_psd_factor(duplicate,[[1.]],weights=[.5]).certified
    assert not certify_psd_factor([[.25]],duplicate).certified
