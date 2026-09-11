import numpy as np
import pytest
from scipy import sparse

import solverpilot as sp


def test_explicit_backend_does_not_discover_unrelated_packages(monkeypatch):
    import solverpilot.runtime.auto as auto
    from solverpilot.backends import ScipyHighsLPBackend
    monkeypatch.setattr(auto, 'default_registry', lambda: pytest.fail('unnecessary discovery'))
    p = sp.LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                                  constraint_lower=[1.], constraint_upper=[np.inf])
    assert auto.solve(p, backend=ScipyHighsLPBackend()).validation.valid


def test_qp_factory_checks_once_and_direct_constructor_cannot_bypass(monkeypatch):
    import solverpilot.problem.quadratic as mod
    original = mod._check_convexity
    calls = []
    def check(P):
        calls.append(1)
        return original(P)
    monkeypatch.setattr(mod, '_check_convexity', check)
    p = sp.QuadraticProblem.from_data(P=[[1.]], A=np.empty((0, 1)), q=[0.],
                                     variable_lower=[-1.], variable_upper=[1.],
                                     constraint_lower=[], constraint_upper=[])
    assert len(calls) == 1
    for switch in (False, True):
        with pytest.raises(ValueError):
            sp.QuadraticProblem(p.linear, sparse.csr_matrix([[-1.]]),
                                convexity_status=sp.ConvexityStatus.CONFIRMED,
                                _verify_convexity=switch)


def test_per_call_tolerance_controls_validation_without_relative_floor():
    from solverpilot.backends import BackendSolveResult, ScipyHighsLPBackend
    class CandidateBackend(ScipyHighsLPBackend):
        def solve(self, p):
            return BackendSolveResult('optimal', np.array([1.-5e-8]), 1.-5e-8)
    p = sp.LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                                  constraint_lower=[1.], constraint_upper=[np.inf])
    assert sp.solve(p, backend=CandidateBackend()).validation.valid
    r = sp.solve(p, backend=CandidateBackend(), tolerances=sp.ValidationTolerances(feasibility=1e-10, feasibility_rel=0.))
    assert not r.validation.valid
    assert r.trace.parameters['validation_tolerances']['feasibility_rel'] == 0.


def test_clarabel_conformance_and_public_gate():
    pytest.importorskip('clarabel')
    from solverpilot.conic import ClarabelBackend, solve_conic
    from solverpilot.conic.clarabel_backend import _runtime_conformance
    assert _runtime_conformance(ClarabelBackend().binding_version)
    m = sp.Model(); t = m.variable(lower=0, upper=5)
    m.psd(t*m.constant(np.eye(2))+m.constant([[0., 1.], [1., 0.]])); m.minimize(t)
    result = solve_conic(m.compile().execution_ir, backend=ClarabelBackend())
    assert result.validated
    assert result.objective_reported == pytest.approx(1., abs=1e-6)


def test_clarabel_psd_symmetry_is_enforced():
    pytest.importorskip('clarabel')
    from solverpilot.conic import ClarabelBackend
    m = sp.Model(); t = m.variable(lower=0, upper=5)
    m.psd(t*m.constant(np.eye(2))+m.constant([[0., 1.], [1., 0.]])); m.minimize(t)
    from dataclasses import replace
    p = m.compile().execution_ir
    p = replace(p, cones=(replace(p.cones[0], g=np.array([0., 1., 2., 0.])),))
    result = ClarabelBackend().solve(p)
    assert not result.validated
    assert result.x is None


def test_independent_kkt_rejects_wrong_dual_and_wrong_primal():
    from solverpilot.validate.optimality import verify_optimality
    p = sp.QuadraticProblem.from_data(P=[[2.]], A=np.empty((0, 1)), q=[-4.],
                                     variable_lower=[0.], variable_upper=[1.],
                                     constraint_lower=[], constraint_upper=[])
    assert verify_optimality(p, [1.], [2.]).verified
    assert not verify_optimality(p, [1.], [1.]).verified
    assert not verify_optimality(p, [2.], [0.]).verified


def test_farkas_separator_is_checked_in_original_coordinates():
    from solverpilot.validate.optimality import verify_infeasibility
    p = sp.LinearProblem.from_data(A=[[1.]], c=[0.], variable_lower=[0.], variable_upper=[1.],
                                  constraint_lower=[2.], constraint_upper=[np.inf])
    assert verify_infeasibility(p, [-1., 1.])
    assert not verify_infeasibility(p, [1., -1.])
    assert not verify_infeasibility(p, [0., 0.])


def test_batch_order_isolation_timeout_and_cancellation():
    from solverpilot.runtime.batch import CancellationToken, solve_batch
    p = sp.LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                                  constraint_lower=[1.], constraint_upper=[np.inf])
    results = solve_batch([p, p], max_workers=2, backend='scipy-highs-ds', timeout_s=30)
    assert [r.index for r in results] == [0, 1]
    assert all(r.validation_valid and r.objective == pytest.approx(1.) for r in results)
    assert solve_batch([p], timeout_s=1e-6)[0].status == 'timeout'
    token = CancellationToken(); token.cancel()
    assert solve_batch([p], cancellation=token)[0].status == 'cancelled'


def test_shared_osqp_instance_serializes_state_and_verifies_duals():
    pytest.importorskip('osqp')
    from concurrent.futures import ThreadPoolExecutor
    from solverpilot.backends import OSQPNativeBackend
    backend = OSQPNativeBackend(eps_abs=1e-9, eps_rel=1e-9)
    def run(q):
        p = sp.QuadraticProblem.from_data(P=[[2.]], A=np.empty((0, 1)), q=[q],
                                         variable_lower=[0.], variable_upper=[1.],
                                         constraint_lower=[], constraint_upper=[])
        return sp.execute(p, backend)
    with ThreadPoolExecutor(max_workers=3) as pool:
        results = list(pool.map(run, [-4., 2., -1.]))
    assert [r.x[0] for r in results] == pytest.approx([1., 0., .5], abs=1e-7)
    assert all(r.optimality_evidence.independently_verified_optimal for r in results)


def test_soft_constraint_and_named_values():
    from solverpilot.model.convenience import indexed_variables, soft_constraint, named_values
    m = sp.Model(); variables = indexed_variables(m, ['a', 'b'], lower=0., upper=1.)
    soft = soft_constraint(m, variables['a'] >= 2., weight=10., name='demand')
    m.minimize(variables['a']+variables['b']+soft.penalty)
    compiled = m.compile(); result = compiled.solve()
    rows = named_values(m, compiled, result)
    assert result.validation.valid
    assert {r['name']: r['value'] for r in rows} == pytest.approx({'x[a]': 1., 'x[b]': 0., 'demand.slack': 1.})


def test_lexicographic_objective_preserves_primary_and_original_model():
    from solverpilot.runtime.multiobjective import solve_lexicographic
    p = sp.LinearProblem.from_data(A=[[1., 1.]], c=[0., 0.], variable_lower=[0., 0.], variable_upper=[2., 2.],
                                  constraint_lower=[1.], constraint_upper=[np.inf])
    before = p.data_hash
    result = solve_lexicographic(p, [[1., 1.], [1., 0.]], tolerance=0.)
    assert result.completed
    assert result.x == pytest.approx([0., 1.], abs=1e-7)
    assert p.data_hash == before


@pytest.mark.parametrize('alpha', [.1, .5, .9])
def test_power_cone_solution_and_exponent_hash(alpha):
    pytest.importorskip('clarabel')
    from solverpilot.model.sets import PowerCone
    from solverpilot.conic import ClarabelBackend
    m = sp.Model(); x = m.variable(3, lower=[4., 2., -10.], upper=[4., 2., 10.])
    m.add_in_set(x, PowerCone(alpha)); m.minimize(-x[2])
    p = m.compile().execution_ir
    result = ClarabelBackend().solve(p)
    assert result.validated
    assert result.objective_reported == pytest.approx(-4**alpha*2**(1-alpha), abs=2e-6)
    from dataclasses import replace
    assert replace(p.cones[0], metadata={'alpha': .3}).structural_hash != p.cones[0].structural_hash


def test_exponential_closure_and_invalid_activity():
    from solverpilot.conic.ir import ConeKind
    from solverpilot.conic.validation import _cone_check
    check = lambda v: _cone_check(ConeKind.EXPONENTIAL, np.array(v), atol=1e-9, rtol=0., source_id=None).valid
    assert check([-1., 0., 0.])
    assert check([1., 1., np.e])
    assert not check([1., 0., 1.])
    assert not check([1., 1., 1.])
    assert not check([np.inf, 1., 1.])


def test_owned_reoptimization_session_transactions_and_rollback():
    pytest.importorskip('osqp')
    from concurrent.futures import ThreadPoolExecutor
    from solverpilot.session.reoptimization import ReoptimizationSession
    m = sp.Model(); x = m.variable(lower=0., upper=3.)
    target = m.parameter(value=1., name='target'); m.minimize((x-target)**2)
    with ReoptimizationSession(m) as session:
        with ThreadPoolExecutor(max_workers=3) as pool:
            results = list(pool.map(lambda v: session.solve(updates={'target': v}), [.5, 1.5, 2.5]))
        assert [r.x[0] for r in results] == pytest.approx([.5, 1.5, 2.5], abs=1e-4)
        assert float(target.value) == 1.
        with pytest.raises(ValueError):
            session.solve(updates={'target': np.nan})
        assert session.solve().validation.valid
    with pytest.raises(RuntimeError, match='closed'):
        session.solve()


def test_running_batch_cancellation_cleans_up_children():
    import multiprocessing
    import threading
    from solverpilot.runtime.batch import CancellationToken, solve_batch
    p = sp.LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                                  constraint_lower=[1.], constraint_upper=[np.inf])
    before = {child.pid for child in multiprocessing.active_children()}
    token = CancellationToken(); timer = threading.Timer(.05, token.cancel)
    timer.start()
    try:
        result = solve_batch([p]*4, max_workers=2, cancellation=token)
    finally:
        timer.cancel(); timer.join()
    assert any(r.status == 'cancelled' for r in result)
    assert {child.pid for child in multiprocessing.active_children()} <= before


def test_named_diagnostics_return_original_constraint_names():
    from solverpilot.model import diagnose_model
    m = sp.Model(); x = m.variable(lower=0., upper=10., name='production')
    m.add(x >= 5., name='minimum-demand'); m.add(x <= 2., name='capacity'); m.minimize(x)
    result = diagnose_model(m)
    names = {r['name'] for r in result['conflict']+result['relaxation']+result['static_issues']}
    assert names & {'minimum-demand', 'capacity'}


def test_batch_memory_limit_is_explicitly_platform_dependent():
    import os
    from solverpilot.runtime.batch import solve_batch
    from solverpilot.exceptions import BudgetNotSupportedError
    if os.name != 'posix':
        with pytest.raises(BudgetNotSupportedError):
            solve_batch([], memory_mb=1024)
    else:
        import resource
        assert hasattr(resource, 'RLIMIT_AS')
        assert solve_batch([], memory_mb=1024) == ()


def test_metadata_cache_can_be_refreshed(monkeypatch):
    from solverpilot.backends import metadata
    calls = []
    monkeypatch.setattr(metadata, '_installed_version', lambda name: calls.append(name) or str(len(calls)))
    metadata.refresh_backend_metadata()
    try:
        assert metadata.version('solver-test') == metadata.version('solver-test') == '1'
        assert calls == ['solver-test']
        metadata.refresh_backend_metadata()
        assert metadata.version('solver-test') == '2'
    finally:
        metadata.refresh_backend_metadata()
