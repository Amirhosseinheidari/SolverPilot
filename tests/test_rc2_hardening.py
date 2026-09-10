import numpy as np
import pytest
from scipy import sparse

import solverpilot as sp
from solverpilot.model import log
from solverpilot.backends import ScipyHighsBackend
from solverpilot.problem import ConvexityStatus, LinearProblem, QuadraticProblem
from solverpilot.validate import CandidateSolution, ValidationTolerances, validate_solution
from solverpilot.cp.ir import (
    CPExactlyOneIR, CPIntVarIR, CPIntervalIR, CPLinearExprIR, CPProblem,
)
from solverpilot.cp.ortools_backend import (
    CP_WORKER_PROTOCOL_VERSION, CP_WORKER_REQUEST_SCHEMA, CP_WORKER_RESULT_SCHEMA,
    ORToolsCPSATBackend, SOLVERPILOT_VERSION, VERIFIED_ORTOOLS_VERSION,
    _bound_matches_integer, _isolated_worker_command, _isolated_worker_environment,
)
from solverpilot.model.errors import CompileError


def _linear(n=2):
    return LinearProblem.from_data(
        A=np.zeros((0,n)), c=np.zeros(n), variable_lower=-np.ones(n), variable_upper=np.ones(n),
        constraint_lower=np.zeros(0), constraint_upper=np.zeros(0),
    )


def test_qp_unknown_convexity_is_fail_closed_at_routing_boundary():
    p=QuadraticProblem(_linear(), sparse.eye(2, format='csr'), ConvexityStatus.UNKNOWN)
    from solverpilot.capabilities import requirements_for
    with pytest.raises(ValueError, match='unverified quadratic'):
        requirements_for(p)


def test_qp_verify_convexity_false_preserves_construction_but_not_routing():
    p=QuadraticProblem.from_data(
        P=np.eye(2), A=np.zeros((0,2)), q=np.zeros(2),
        variable_lower=[-1,-1], variable_upper=[1,1],
        constraint_lower=[], constraint_upper=[], verify_convexity=False,
    )
    assert p.convexity_status is ConvexityStatus.UNKNOWN
    from solverpilot.capabilities import requirements_for
    with pytest.raises(ValueError, match='unverified quadratic'):
        requirements_for(p)


def test_qp_nonconvex_cannot_be_smuggled_into_convex_runtime():
    with pytest.raises(ValueError, match='positive semidefinite'):
        QuadraticProblem.from_data(
            P=np.diag([1.0,-1.0]), A=np.zeros((0,2)), q=np.zeros(2),
            variable_lower=[-1,-1], variable_upper=[1,1], constraint_lower=[], constraint_upper=[],
        )


def test_scaled_validation_accepts_tiny_relative_residual_at_large_scale():
    p=LinearProblem.from_data(
        A=[[1e12]], c=[0.0], variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[1e12], constraint_upper=[1e12],
    )
    x=np.array([1.000000000000001])
    r=validate_solution(p,CandidateSolution(x))
    assert r.max_constraint_violation > 1e-4
    assert r.valid
    assert any("scale-aware" in w for w in r.warnings)


def test_cp_problem_rejects_unknown_interval_variable_reference():
    v=CPIntVarIR(1,'x',(0,1),'int')
    with pytest.raises(ValueError, match='unknown CP variable'):
        CPProblem((v,), (CPIntervalIR(1,'bad',99,1,1),), ())


def test_cp_problem_rejects_non_boolean_exactly_one_literal():
    v=CPIntVarIR(1,'x',(0,1,2),'int')
    with pytest.raises(ValueError, match='requires Boolean'):
        CPProblem((v,), (), (CPExactlyOneIR((1,)),))


def test_cp_problem_rejects_unknown_objective_variable():
    v=CPIntVarIR(1,'x',(0,1),'bool')
    with pytest.raises(ValueError, match='objective references unknown'):
        CPProblem((v,), (), (), objective=CPLinearExprIR(((2,1),)))


def test_cp_controller_timeout_must_be_positive():
    with pytest.raises(ValueError, match='controller_timeout_s'):
        ORToolsCPSATBackend(controller_timeout_s=0)


def test_nlp_log_domain_must_be_provable_from_variable_bounds():
    m=sp.Model(); x=m.variable(lower=-1, upper=2); m.minimize(log(x))
    from solverpilot.nlp.compiler import compile_nlp_model
    c=compile_nlp_model(m)
    assert c.execution_ir.metadata['domain_hazards']
    assert not c.execution_ir.metadata['domain_safety_proven']


def test_nlp_log_positive_domain_compiles():
    m=sp.Model(); x=m.variable(lower=0.1, upper=2); m.minimize(log(x))
    from solverpilot.nlp.compiler import compile_nlp_model
    c=compile_nlp_model(m)
    assert c.execution_ir.n_variables == 1
    assert c.execution_ir.metadata['domain_safety_proven']


def test_nlp_division_denominator_may_not_cross_zero():
    m=sp.Model(); x=m.variable(lower=-1,upper=1); y=m.variable(lower=-1,upper=1); m.minimize(x/y)
    from solverpilot.nlp.compiler import compile_nlp_model
    c=compile_nlp_model(m)
    assert any(h['kind']=='div' for h in c.execution_ir.metadata['domain_hazards'])


def test_runtime_exposes_optimality_trust_separate_from_legacy_status():
    p=LinearProblem.from_data(A=[[1.0]],c=[1.0],variable_lower=[0.0],variable_upper=[1.0],constraint_lower=[0.0],constraint_upper=[np.inf])
    r=sp.execute(p,ScipyHighsBackend())
    assert r.status is sp.PublicStatus.VALID_OPTIMAL
    assert r.optimality_evidence.backend_reported_optimal
    assert r.optimality_evidence.primal_validated
    assert not r.optimality_evidence.independently_verified_optimal
    assert r.raw_statistics['solverpilot_trust']['primal_validated'] is True

def test_mps_multiple_rhs_vectors_fail_closed(tmp_path):
    from solverpilot.problem.mps import MPSUnsupportedFeatureError, read_mps
    text='''NAME X\nROWS\n N OBJ\n E R1\nCOLUMNS\n X R1 1\nRHS\n RHS1 R1 1\n RHS2 R1 2\nENDATA\n'''
    path=tmp_path/'multi.mps'; path.write_text(text)
    with pytest.raises(MPSUnsupportedFeatureError,match='multiple RHS vectors'):
        read_mps(path)


def test_scaled_validation_does_not_hide_large_absolute_error_near_zero_rhs():
    p=LinearProblem.from_data(
        A=[[1e12]], c=[0.0], variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[0.0], constraint_upper=[0.0],
    )
    # Activity is 1000, so this must remain invalid even though A itself is huge.
    r=validate_solution(p,CandidateSolution(np.array([1e-9])))
    assert r.max_constraint_violation == pytest.approx(1000.0)
    assert not r.valid


def _mock_worker_response(call_kwargs, *, assignment, objective, optimality_proven, status='optimal', raw_statistics=None, **overrides):
    import json
    request=json.loads(call_kwargs['input'])
    raw={'ortools_version':request['ortools_version'],'model_validation':''}
    if raw_statistics is not None: raw.update(raw_statistics)
    payload={
        'schema':CP_WORKER_RESULT_SCHEMA,'protocol_version':CP_WORKER_PROTOCOL_VERSION,
        'request_id':request['request_id'],'solverpilot_version':request['solverpilot_version'],
        'ortools_version':request['ortools_version'],'problem_structural_hash':request['problem_structural_hash'],
        'problem_data_hash':request['problem_data_hash'],'status':status,'assignment':assignment,
        'objective':objective,'optimality_proven':optimality_proven,'raw_statistics':raw,
    }
    payload.update(overrides); return payload


def test_cp_num_workers_must_be_positive_integer():
    for bad in (0,-1,True,False,1.5,'2'):
        with pytest.raises(ValueError, match='num_workers'): ORToolsCPSATBackend(num_workers=bad)
    assert ORToolsCPSATBackend(num_workers=1).num_workers == 1


def test_cp_worker_launch_is_isolated_and_sanitized(monkeypatch):
    import json, os, subprocess
    from types import SimpleNamespace
    m=sp.CPModel(); x=m.bool_var('x'); m.maximize(x); p=m.compile(); backend=ORToolsCPSATBackend()
    monkeypatch.setattr(ORToolsCPSATBackend,'_require',lambda self:None); monkeypatch.setenv('PYTHONPATH','/tmp/attacker'); monkeypatch.setenv('PYTHONHOME','/tmp/home')
    seen={}
    def fake_run(*a,**k):
        seen.update(command=a[0],cwd=k.get('cwd'),env=k.get('env')); payload=_mock_worker_response(k,assignment={str(x.var_id):1},objective=1,optimality_proven=True,raw_statistics={'status':'OPTIMAL','solver_objective_value':1.0,'best_bound':1.0})
        return SimpleNamespace(returncode=0,stdout=json.dumps(payload),stderr='')
    monkeypatch.setattr(subprocess,'run',fake_run); r=backend.solve(p)
    assert r.validation.valid and r.optimality_proven; assert '-I' in seen['command']; assert seen['cwd'] and seen['cwd'] != os.getcwd(); assert 'PYTHONPATH' not in seen['env']; assert 'PYTHONHOME' not in seen['env']


def test_cp_parent_revalidation_can_revoke_worker_optimality(monkeypatch):
    import json, subprocess
    from types import SimpleNamespace
    from solverpilot.cp.ir import CPLinearConstraintIR
    p=CPProblem((CPIntVarIR(1,'x',(0,1),'bool'),),(),(CPLinearConstraintIR(CPLinearExprIR(((1,1),)),lower=1,upper=1),)); backend=ORToolsCPSATBackend(); monkeypatch.setattr(ORToolsCPSATBackend,'_require',lambda self:None)
    def fake_run(*a,**k): return SimpleNamespace(returncode=0,stdout=json.dumps(_mock_worker_response(k,assignment={'1':0},objective=None,optimality_proven=True,raw_statistics={'status':'OPTIMAL'})),stderr='')
    monkeypatch.setattr(subprocess,'run',fake_run); r=backend.solve(p); assert not r.validation.valid; assert not r.optimality_proven


@pytest.mark.parametrize('field',['request_id','solverpilot_version','ortools_version','problem_structural_hash','problem_data_hash'])
def test_cp_response_binding_mismatch_rejected(monkeypatch, field):
    import json, subprocess
    from types import SimpleNamespace
    m=sp.CPModel(); x=m.bool_var('x'); m.maximize(x); p=m.compile(); backend=ORToolsCPSATBackend(); monkeypatch.setattr(ORToolsCPSATBackend,'_require',lambda self:None)
    def fake_run(*a,**k):
        payload=_mock_worker_response(k,assignment={str(x.var_id):1},objective=1,optimality_proven=True,raw_statistics={'status':'OPTIMAL','solver_objective_value':1.0,'best_bound':1.0}); payload[field]='tampered'; return SimpleNamespace(returncode=0,stdout=json.dumps(payload),stderr='')
    monkeypatch.setattr(subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='binding mismatch'): backend.solve(p)


def test_cp_worker_proof_bit_is_strict_boolean(monkeypatch):
    import json, subprocess
    from types import SimpleNamespace
    m=sp.CPModel(); x=m.bool_var('x'); m.maximize(x); p=m.compile(); backend=ORToolsCPSATBackend(); monkeypatch.setattr(ORToolsCPSATBackend,'_require',lambda self:None)
    def fake_run(*a,**k): return SimpleNamespace(returncode=0,stdout=json.dumps(_mock_worker_response(k,assignment={str(x.var_id):1},objective=1,optimality_proven='false',raw_statistics={'status':'OPTIMAL','solver_objective_value':1.0,'best_bound':1.0})),stderr='')
    monkeypatch.setattr(subprocess,'run',fake_run)
    with pytest.raises(RuntimeError,match='non-boolean optimality_proven'): backend.solve(p)


def test_cp_valid_primal_needs_bound_consistency_for_proof(monkeypatch):
    import json, subprocess
    from types import SimpleNamespace
    m=sp.CPModel(); x=m.bool_var('x'); m.maximize(x); p=m.compile(); backend=ORToolsCPSATBackend(); monkeypatch.setattr(ORToolsCPSATBackend,'_require',lambda self:None)
    def fake_run(*a,**k): return SimpleNamespace(returncode=0,stdout=json.dumps(_mock_worker_response(k,assignment={str(x.var_id):1},objective=1,optimality_proven=True,raw_statistics={'status':'OPTIMAL','solver_objective_value':1.0,'best_bound':0.0})),stderr='')
    monkeypatch.setattr(subprocess,'run',fake_run); r=backend.solve(p); assert r.validation.valid; assert not r.optimality_proven


def test_cp_large_integer_float_bound_proof_fails_closed():
    assert _bound_matches_integer(float(2**53),2**53)
    assert not _bound_matches_integer(float(2**53+2),2**53+2)


def test_cp_isolated_interpreter_resists_shadowing(tmp_path):
    import json, subprocess
    shadow=tmp_path/'solverpilot'; shadow.mkdir(); (shadow/'__init__.py').write_text("raise RuntimeError('SHADOW PACKAGE LOADED')\n")
    m=sp.CPModel(); x=m.bool_var('x'); p=m.compile(); request={'schema':CP_WORKER_REQUEST_SCHEMA,'protocol_version':CP_WORKER_PROTOCOL_VERSION,'request_id':'0'*32,'solverpilot_version':SOLVERPILOT_VERSION,'ortools_version':VERIFIED_ORTOOLS_VERSION,'problem_structural_hash':p.structural_hash,'problem_data_hash':p.data_hash,'problem':p.canonical_dict(),'max_time_s':None,'num_workers':1}
    env=_isolated_worker_environment(); env['PYTHONPATH']=str(tmp_path); proc=subprocess.run(_isolated_worker_command(),input=json.dumps(request),text=True,capture_output=True,cwd=tmp_path,env=env,timeout=30)
    assert 'SHADOW PACKAGE LOADED' not in proc.stderr
    assert proc.returncode in (0,2)
