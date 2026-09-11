"""Bounded, process-isolated LP/QP batch execution.

The caller must use the normal multiprocessing main guard in scripts. Each job
owns its solver; deadlines include process startup and cleanup. IPC is private
to child processes created here, never an interchange format for external data.
"""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import multiprocessing as mp
import os
import threading
from time import monotonic

import numpy as np
from solverpilot._immutability import readonly_array
from solverpilot.exceptions import BudgetNotSupportedError
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.validate import CandidateSolution, ValidationTolerances, validate_solution


class CancellationToken:
    def __init__(self):
        self._event = threading.Event()

    def cancel(self):
        self._event.set()

    @property
    def cancelled(self):
        return self._event.is_set()


@dataclass(frozen=True, slots=True)
class BatchItem:
    index: int
    status: str
    x: np.ndarray | None
    objective: float | None
    validation_valid: bool
    backend: str | None
    elapsed_s: float
    error: str | None = None

    def __post_init__(self):
        if self.x is not None:
            object.__setattr__(self, 'x', readonly_array(self.x, dtype=float))


def _worker(connection, problem, backend, tolerances, memory_mb, solver_budget):
    try:
        if memory_mb is not None:
            import resource
            limit = int(memory_mb*1024*1024)
            resource.setrlimit(resource.RLIMIT_AS, (limit, limit))
        from solverpilot.runtime import solve
        result = solve(problem, backend=backend, tolerances=tolerances, budget=solver_budget)
        connection.send({'status': result.status.value,
                         'x': None if result.x is None else result.x.tolist(),
                         'objective': result.objective, 'backend': result.trace.backend})
    except BaseException as exc:
        connection.send({'error': f'{type(exc).__name__}: {exc}'})
    finally:
        connection.close()


def solve_batch(problems, *, backend: str | None = None, max_workers=1,
                timeout_s=None, cancellation: CancellationToken | None = None,
                memory_mb=None, tolerances: ValidationTolerances | None = None, solver_budget=None):
    """Return ordered results; timeout is per job, measured before process start.

    Cancellation/timeouts terminate the worker and do not invent an incumbent.
    Memory is an address-space cap, supported only on POSIX. It is not an RSS cap.
    ``max_workers`` bounds live solver processes, not their internal thread count.
    """
    if isinstance(max_workers, bool) or not isinstance(max_workers, int) or max_workers < 1:
        raise ValueError('max_workers must be a positive integer')
    for name, value in [('timeout_s', timeout_s), ('memory_mb', memory_mb)]:
        if value is not None and (isinstance(value, bool) or not np.isfinite(value) or value <= 0):
            raise ValueError(f'{name} must be finite and positive')
    if memory_mb is not None and os.name != 'posix':
        raise BudgetNotSupportedError('batch address-space limits require POSIX RLIMIT_AS')
    if backend is not None and not isinstance(backend, str):
        raise TypeError('batch backend must be a registered built-in name; mutable instances cannot be shared')
    jobs = tuple(problems)
    if any(not isinstance(p, (LinearProblem, QuadraticProblem)) for p in jobs):
        raise TypeError('batch currently accepts canonical LinearProblem and QuadraticProblem')
    token = cancellation or CancellationToken()
    tol = tolerances or ValidationTolerances()
    ctx = mp.get_context('spawn')

    def run(pair):
        index, problem = pair
        start = monotonic()
        def stopped(status, error=None):
            return BatchItem(index, status, None, None, False, None, monotonic()-start, error)
        if token.cancelled:
            return stopped('cancelled')
        parent, child = ctx.Pipe(duplex=False)
        proc = ctx.Process(target=_worker, args=(child, problem, backend, tol, memory_mb, solver_budget))
        try:
            proc.start(); child.close()
            while True:
                if token.cancelled:
                    return stopped('cancelled')
                if timeout_s is not None and monotonic()-start >= timeout_s:
                    return stopped('timeout')
                if parent.poll(.01):
                    try:
                        payload = parent.recv()
                    except EOFError:
                        return stopped('error', 'worker closed without a result')
                    if 'error' in payload:
                        return stopped('error', payload['error'])
                    x = payload['x']
                    validation = None if x is None else validate_solution(problem, CandidateSolution(np.asarray(x), payload['objective']), tolerances=tol)
                    valid = validation is not None and validation.valid
                    status = payload['status'] if x is None or valid else 'invalid_solution'
                    return BatchItem(index, status, None if x is None else np.asarray(x),
                                     None if validation is None else validation.objective_recomputed,
                                     valid, payload['backend'], monotonic()-start)
                if not proc.is_alive():
                    return stopped('error', f'worker exited with code {proc.exitcode}')
        finally:
            child.close(); parent.close()
            if proc.pid is not None:
                if proc.is_alive():
                    proc.terminate()
                proc.join(timeout=2)
                if proc.is_alive():
                    proc.kill(); proc.join()
                proc.close()

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return tuple(pool.map(run, enumerate(jobs)))
