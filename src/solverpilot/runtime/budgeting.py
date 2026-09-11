from __future__ import annotations

from dataclasses import is_dataclass, replace

from solverpilot.backends import Backend
from solverpilot.plan import SolveBudget
from solverpilot.exceptions import BudgetNotSupportedError


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
