"""Common options mapped only to controls with defined backend semantics."""

from solverpilot.exceptions import BudgetNotSupportedError
from .budgeting import configured_backend


def core_controls(kwargs):
    kwargs = dict(kwargs)
    if kwargs.pop("progress", None) is not None or kwargs.pop("cancellation", None) is not None:
        raise BudgetNotSupportedError("LP/QP progress/cancellation requires the batch API")
    return kwargs


def nlp_controls(backend, kwargs):
    kwargs = dict(kwargs)
    if isinstance(backend, str):
        from .catalog import backend_catalog

        backend = backend_catalog()[backend]
    if kwargs.pop("progress", None) is not None or kwargs.pop("cancellation", None) is not None:
        raise BudgetNotSupportedError(
            "NLP iteration callbacks are not implemented; use a native time budget"
        )
    budget = kwargs.pop("budget", None)
    if budget is not None:
        if budget.memory_mb is not None or budget.threads is not None:
            raise BudgetNotSupportedError("NLP memory/thread budgets are not implemented")
        if budget.wall_time_s is not None:
            backend = configured_backend(backend, {"time_limit_s": budget.wall_time_s})
    return backend, kwargs


def minlp_controls(kwargs):
    kwargs = dict(kwargs)
    budget = kwargs.pop("budget", None)
    if (
        budget is not None
        or kwargs.pop("progress", None) is not None
        or kwargs.pop("cancellation", None) is not None
    ):
        raise BudgetNotSupportedError(
            "MINLP aggregate resource/callback controls are not implemented; max_iter bounds OA iterations"
        )
    tol = kwargs.pop("tolerances", None)
    if tol is not None:
        if "atol" in kwargs:
            raise ValueError("atol and tolerances supplied together")
        kwargs["atol"] = tol.feasibility
    return kwargs
