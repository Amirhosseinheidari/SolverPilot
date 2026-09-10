from __future__ import annotations

from dataclasses import is_dataclass, replace

from solverpilot.backends import Backend
from solverpilot.plan import SolveBudget
from solverpilot.exceptions import BudgetNotSupportedError


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
    return replace(backend, **updates)
