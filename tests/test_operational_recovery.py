"""Lifecycle contracts executed by the full OS/Python CI matrix."""
import threading
import pytest
import numpy as np

from solverpilot.runtime.batch_stream import BatchExecutor, iter_solve_batch
from solverpilot.runtime.batch import CancellationToken
from test_total_deadline import problem


def test_timeout_then_restart_in_same_executor():
    with BatchExecutor(backend='scipy-highs-ds',max_workers=1,max_pending=1,timeout_s=.00001) as executor:
        result=list(executor.iter([problem()]))[0]
        assert result.status=='timeout' and result.x is None
        executor.timeout_s=30
        result=list(executor.iter([problem()]))[0]
        assert result.validation_valid and result.independently_verified_optimal


def test_active_cancellation_then_fresh_request():
    token=CancellationToken()
    with BatchExecutor(backend='scipy-highs-ds',max_workers=1,max_pending=1,timeout_s=30) as executor:
        timer=threading.Timer(.001,token.cancel)
        timer.start()
        try: result=list(executor.iter([problem()],cancellation=token))[0]
        finally: timer.join()
        assert result.status=='cancelled' and result.x is None
        assert list(executor.iter([problem()]))[0].independently_verified_optimal


def test_worker_crash_does_not_poison_subsequent_request():
    with BatchExecutor(backend='scipy-highs-ds',max_workers=1,max_pending=1,timeout_s=30) as executor:
        assert list(executor.iter([problem()]))[0].validation_valid
        worker=executor._workers[0]
        worker.process.terminate();worker.process.join()
        assert list(executor.iter([problem()]))[0].status=='error'
        assert list(executor.iter([problem()]))[0].independently_verified_optimal


def test_sequential_batch_preserves_independent_evidence():
    item=list(iter_solve_batch([problem()],mode='sequential',backend='scipy-highs-ds'))[0]
    assert item.independently_verified_optimal and item.problem_data_hash==problem().data_hash


def test_builtin_osqp_first_call_is_as_valid_as_warm_calls():
    pytest.importorskip('osqp')
    from solverpilot import QuadraticProblem, solve
    from solverpilot.runtime.auto import default_registry
    p=QuadraticProblem.from_data(P=np.eye(2),q=[-1.,-1.],A=[[1.,1.]],
        variable_lower=[0.,0.],variable_upper=[2.,2.],constraint_lower=[0.],constraint_upper=[1.])
    registry=default_registry()
    for _ in range(3):
        result=solve(p,backend='osqp-native',registry=registry)
        assert result.validation.valid and result.optimality_evidence.independently_verified_optimal
        assert result.objective==pytest.approx(-.75)
