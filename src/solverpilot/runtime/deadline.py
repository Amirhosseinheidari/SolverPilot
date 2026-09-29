"""Explicit deadline-controlled CPU LP/QP call using the existing batch worker.

Use the normal multiprocessing main guard. Worker startup, solve, proof checks,
transport and parent validation count; process cleanup can add latency.
Only registered in-process CPU adapters are eligible, avoiding orphaned nested
workers. A late result is discarded, never promoted to on-time success.
"""
from dataclasses import replace
from time import monotonic
import math

from .batch_stream import BatchExecutor


def solve_with_deadline(problem, *, timeout_s, backend=None, tolerances=None, cancellation=None):
    if isinstance(timeout_s, bool) or not math.isfinite(timeout_s) or timeout_s <= 0:
        raise ValueError('timeout_s must be finite and positive')
    allowed = {None, 'highspy-native', 'scipy-highs-ds', 'scipy-highs-ipm',
               'scipy-highs-bridge', 'osqp-native'}
    if backend not in allowed:
        raise ValueError('deadline calls require a supported in-process CPU backend name')
    from solverpilot.plan import SolveBudget
    start = monotonic()
    with BatchExecutor(backend=backend, max_workers=1, max_pending=1, timeout_s=timeout_s,
                       tolerances=tolerances, solver_budget=SolveBudget(wall_time_s=timeout_s)) as executor:
        result = next(executor.iter([problem], cancellation=cancellation))
    elapsed = monotonic()-start
    if elapsed > timeout_s and result.status not in {'timeout', 'cancelled', 'error'}:
        return replace(result, status='timeout', x=None, objective=None, validation_valid=False,
                       independently_verified_optimal=False, elapsed_s=elapsed,
                       requested_time_s=timeout_s, within_budget=False,
                       error='whole-call deadline exceeded; late result discarded')
    return replace(result, elapsed_s=elapsed, requested_time_s=timeout_s, within_budget=elapsed <= timeout_s)
