"""Shared solve controls, with explicit errors for unsupported guarantees."""

from dataclasses import dataclass, field
from typing import Callable, Any
from solverpilot.plan import SolveBudget
from solverpilot.validate import ValidationTolerances


@dataclass(frozen=True, slots=True)
class ProgressEvent:
    backend: str
    phase: str
    iteration: int | None = None
    elapsed_s: float | None = None
    objective: float | None = None
    primal_residual: float | None = None
    dual_residual: float | None = None


@dataclass(frozen=True, slots=True)
class SolveOptions:
    budget: SolveBudget | None = None
    tolerances: ValidationTolerances = field(default_factory=ValidationTolerances)
    progress: Callable[[ProgressEvent], bool | None] | None = None
    cancellation: Any = None

    def __post_init__(self):
        if self.budget is not None and not isinstance(self.budget, SolveBudget):
            raise TypeError("budget must be SolveBudget")
        if not isinstance(self.tolerances, ValidationTolerances):
            raise TypeError("tolerances must be ValidationTolerances")
        if self.progress is not None and not callable(self.progress):
            raise TypeError("progress must be callable")
        if self.cancellation is not None and not hasattr(self.cancellation, "cancelled"):
            raise TypeError("cancellation must provide a cancelled property")


def expand_options(kwargs):
    kwargs = dict(kwargs)
    options = kwargs.pop("options", None)
    if options is not None:
        if not isinstance(options, SolveOptions):
            raise TypeError("options must be SolveOptions")
        for name in ("budget", "tolerances", "progress", "cancellation"):
            value = getattr(options, name)
            if value is not None:
                if name in kwargs:
                    raise ValueError(f"{name} supplied twice")
                kwargs[name] = value
    return kwargs
