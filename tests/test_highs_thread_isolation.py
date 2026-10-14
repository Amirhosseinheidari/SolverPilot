from pathlib import Path
import os
import subprocess
import sys
from types import SimpleNamespace

import pytest

from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.backends import HighspyNativeBackend
from solverpilot.backends._highs_thread_transport import read_problem, write_problem, solve_isolated_threads
from time import perf_counter


def problem():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
        constraint_lower=[1.], constraint_upper=[float('inf')])


@pytest.mark.parametrize('threads', [True, 0, -1, .5, 1.5, '2'])
def test_threads_must_be_positive_integers(threads):
    with pytest.raises(ValueError, match='positive integer'):
        HighspyNativeBackend(threads=threads)._options()


def test_model_transport_preserves_domains_sense_offset_and_quadratic_data(tmp_path):
    cases = [LinearProblem.from_data(A=[[2.]], c=[3.], variable_lower=[0.], variable_upper=[4.],
        constraint_lower=[0.], constraint_upper=[8.], domains=['integer'], objective_sense='maximize', objective_offset=7.),
        QuadraticProblem.from_data(A=[[1.]], P=[[2.]], q=[-3.], variable_lower=[0.], variable_upper=[2.],
            constraint_lower=[0.], constraint_upper=[2.], objective_offset=5.)]
    for p in cases:
        path = tmp_path/'model.npz'
        write_problem(path, p)
        assert read_problem(path).data_hash == p.data_hash


def test_threads_can_change_without_resetting_any_global_scheduler():
    pytest.importorskip('highspy')
    code = '''
import highspy
from concurrent.futures import ThreadPoolExecutor
from solverpilot import LinearProblem, QuadraticProblem, execute
from solverpilot.backends import HighspyNativeBackend
highspy.Highs.resetGlobalScheduler = lambda *_: (_ for _ in ()).throw(AssertionError('global reset'))
p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.], constraint_lower=[1.], constraint_upper=[float('inf')])
b = HighspyNativeBackend(reuse=True, solver='simplex', threads=1)
assert execute(p,b).objective == 1.
b.threads=2
r=execute(p,b)
assert r.objective == 1. and r.optimality_evidence.independently_verified_optimal
assert r.raw_statistics['isolated_thread_configuration'] and not r.trace.reuse_applied
b.threads=1
assert execute(p,b).objective == 1.
c = HighspyNativeBackend(threads=2, solver='simplex')
with ThreadPoolExecutor(2) as pool:
    results=list(pool.map(lambda backend: execute(p,backend), (b,c)))
assert all(r.objective == 1. and r.optimality_evidence.independently_verified_optimal for r in results)
assert execute(p,b).trace.reuse_applied
integer = LinearProblem.from_data(A=[[2.]], c=[3.], variable_lower=[0.], variable_upper=[4.], constraint_lower=[0.], constraint_upper=[8.], domains=['integer'], objective_sense='maximize', objective_offset=7.)
qp = QuadraticProblem.from_data(A=[[1.]], P=[[2.]], q=[-3.], variable_lower=[0.], variable_upper=[2.], constraint_lower=[0.], constraint_upper=[2.], objective_offset=5.)
for target, objective in ((integer,19.), (qp,2.75)):
    r=execute(target,c)
    assert abs(r.objective-objective) < 1e-6 and r.validation.valid
    assert r.raw_statistics['isolated_thread_configuration']
b.close(); c.close()
'''
    env = dict(os.environ, PYTHONPATH=str(Path(__file__).resolve().parents[1]/'src'))
    completed = subprocess.run([sys.executable, '-c', code], capture_output=True, text=True, env=env, timeout=30)
    assert completed.returncode == 0, completed.stderr


def test_generic_native_error_does_not_trigger_isolation(monkeypatch):
    highspy = pytest.importorskip('highspy')
    monkeypatch.setattr(highspy.Highs, 'run', lambda _: highspy.HighsStatus.kError)
    monkeypatch.setattr('solverpilot.backends._highs_thread_transport.solve_isolated_threads',
                        lambda *a, **k: pytest.fail('generic error retried'))
    with pytest.raises(RuntimeError, match='run failed'):
        HighspyNativeBackend(threads=1).solve(problem())


def test_isolation_charges_preparation_and_rejects_timeout(monkeypatch):
    b = HighspyNativeBackend(threads=2, time_limit_s=.001)
    r = solve_isolated_threads(problem(), b, started=perf_counter()-1.)
    assert r.x is None and r.backend_status == 'limit_no_solution'
    def timeout(*a, **k):
        raise subprocess.TimeoutExpired('owned-worker', .001)
    monkeypatch.setattr(subprocess, 'run', timeout)
    b.time_limit_s=1.
    r = solve_isolated_threads(problem(), b, started=perf_counter())
    assert r.x is None and r.backend_status == 'limit_no_solution'


def test_isolation_rejects_mismatched_transport_identity(monkeypatch):
    import json
    from solverpilot.backends._highs_thread_transport import PROTOCOL, PREFIX
    reply = dict(protocol=PROTOCOL, problem_data_hash='a'*64, status='optimal', x=[1.],
                 objective=1., statistics={})
    monkeypatch.setattr(subprocess, 'run', lambda *a, **k: SimpleNamespace(
        returncode=0, stdout=PREFIX+json.dumps(reply), stderr=''))
    with pytest.raises(RuntimeError, match='invalid isolated'):
        solve_isolated_threads(problem(), HighspyNativeBackend(threads=2), started=perf_counter())
