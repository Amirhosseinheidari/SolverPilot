from __future__ import annotations

from dataclasses import is_dataclass, replace
from time import perf_counter

from solverpilot.backends import Backend
from solverpilot.plan import SolveBudget
from solverpilot.exceptions import BudgetNotSupportedError


class CallBudget:
    """Cooperative aggregate accounting, including setup and validation."""
    def __init__(self, budget=None):
        if budget is not None and not isinstance(budget, SolveBudget): raise TypeError('budget must be SolveBudget')
        self.budget=budget; self.started=perf_counter()

    def remaining_s(self):
        return None if self.budget is None or self.budget.wall_time_s is None else self.budget.wall_time_s-(perf_counter()-self.started)

    def remaining_budget(self):
        left=self.remaining_s()
        if left is not None and left <= 0: raise TimeoutError('aggregate solve budget exhausted')
        return self.budget if left is None else replace(self.budget,wall_time_s=left)

    def report(self):
        elapsed=perf_counter()-self.started
        requested=None if self.budget is None else self.budget.wall_time_s
        return {'requested_s':requested,'elapsed_s':elapsed,'within_budget':None if requested is None else elapsed<=requested,
                'budget_exhausted':requested is not None and elapsed>=requested,'enforcement':'native_soft_limit; aggregate elapsed includes setup and validation'}


class _ConfiguredBackend:
    """Apply call-local settings while preserving an owned, locked workspace."""
    def __init__(self, backend, updates):
        self.backend = backend
        self.updates = dict(updates)

    def __getattr__(self, name):
        return self.updates[name] if name in self.updates else getattr(self.backend, name)

    def solve(self, problem, **kwargs):
        with self.backend._lock:
            previous = {key: getattr(self.backend, key) for key in self.updates}
            try:
                for key, value in self.updates.items():
                    setattr(self.backend, key, value)
                return self.backend.solve(problem, **kwargs)
            finally:
                for key, value in previous.items():
                    setattr(self.backend, key, value)


def configured_backend(backend, updates):
    if not updates:
        return backend
    if hasattr(backend, '_lock'):
        return _ConfiguredBackend(backend, updates)
    if not is_dataclass(backend):
        raise TypeError('backend options require a dataclass or an owned reentrant lock')
    return replace(backend, **updates)


def apply_budget(backend: Backend, budget: SolveBudget | None) -> Backend:
    if budget is None:
        return backend
    if budget.memory_mb is not None:
        raise BudgetNotSupportedError(
            "memory budgets are modeled but are not yet enforceable by built-in backends"
        )

    updates: dict[str, object] = {}
    if budget.wall_time_s is not None:
        if backend.manifest.metadata.get("supports_wall_time_budget") is False:
            raise BudgetNotSupportedError(
                f"backend {backend.manifest.name!r} has no verified wall-time budget mapping"
            )
        if not hasattr(backend, "time_limit_s"):
            raise BudgetNotSupportedError(
                f"backend {backend.manifest.name!r} cannot enforce a wall-time budget"
            )
        updates["time_limit_s"] = float(budget.wall_time_s)

    if budget.threads is not None:
        if not hasattr(backend, "threads"):
            raise BudgetNotSupportedError(
                f"backend {backend.manifest.name!r} cannot enforce a thread budget"
            )
        updates["threads"] = int(budget.threads)

    if not updates:
        return backend
    if not is_dataclass(backend):
        raise TypeError(
            f"backend {backend.manifest.name!r} exposes budget fields but is not safely cloneable"
        )
    return configured_backend(backend, updates)
