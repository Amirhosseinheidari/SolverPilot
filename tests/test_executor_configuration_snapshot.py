"""Temporary settings from concurrent calls must not leak into replay traces."""
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from threading import Event, Lock, RLock

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot._synchronization import serialized
from solverpilot.backends.base import BackendSolveResult
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.plan import SolveBudget
from solverpilot.runtime import executor
from solverpilot.runtime.budgeting import apply_budget


@dataclass
class ConcurrentBackend:
    time_limit_s: float | None = None
    threads: int | None = 1
    _lock: object = field(default_factory=RLock, init=False)
    _second_solving: Event = field(default_factory=Event, init=False)
    _release_second: Event = field(default_factory=Event, init=False)

    manifest = BackendManifest(name="concurrent", capabilities={Capability.LP: SupportLevel.NATIVE})

    @property
    def _execution_lock(self):
        return self._lock

    def is_available(self):
        return True

    @serialized
    def solve(self, problem):
        actual_limit = self.time_limit_s
        if problem.c[0] == 2:
            self._second_solving.set()
            assert self._release_second.wait(10)
        return BackendSolveResult("optimal", np.array([0.]), 0., {"actual_limit": actual_limit})


@pytest.mark.parametrize("first_budget", [None, SolveBudget(threads=1)])
def test_settings_snapshot_precedes_other_calls_temporary_budget(monkeypatch, first_budget):
    backend = ConcurrentBackend()
    validating_first = Event()
    validate = executor.validate_solution

    def interleaved_validation(problem, candidate, **kwargs):
        if problem.c[0] == 1:
            validating_first.set()
            assert backend._second_solving.wait(10)
        return validate(problem, candidate, **kwargs)

    monkeypatch.setattr(executor, "validate_solution", interleaved_validation)
    data = dict(A=[[1.]], variable_lower=[0.], variable_upper=[2.],
                constraint_lower=[0.], constraint_upper=[2.])
    first_problem = LinearProblem.from_data(c=[1.], **data)
    second_problem = LinearProblem.from_data(c=[2.], **data)
    with ThreadPoolExecutor(max_workers=2) as pool:
        first = pool.submit(executor.execute, first_problem, apply_budget(backend, first_budget))
        try:
            assert validating_first.wait(10)
            second = pool.submit(executor.execute, second_problem,
                                 apply_budget(backend, SolveBudget(wall_time_s=2)))
            result = first.result(timeout=10)
            assert result.validation.valid
            assert result.trace.parameters["backend_configuration"]["time_limit_s"] is None
            assert result.raw_statistics["actual_limit"] is None
        finally:
            backend._release_second.set()
        result2 = second.result(timeout=10)
    assert result2.trace.parameters["backend_configuration"]["time_limit_s"] == 2
    assert result2.raw_statistics["actual_limit"] == 2
    assert backend.time_limit_s is None


def test_third_party_private_lock_is_not_assumed_to_be_reentrant():
    class PrivateLockBackend:
        manifest = ConcurrentBackend.manifest

        def __init__(self):
            self._lock = Lock()

        def is_available(self):
            return True

        def solve(self, problem):
            # A bounded acquire exposes regressions without hanging the suite.
            if not self._lock.acquire(timeout=.1):
                raise RuntimeError('executor acquired a third-party private lock')
            try:
                return BackendSolveResult('optimal', np.array([0.]), 0.)
            finally:
                self._lock.release()

    problem = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
                                      variable_upper=[2.], constraint_lower=[0.], constraint_upper=[2.])
    assert executor.execute(problem, PrivateLockBackend()).validation.valid
