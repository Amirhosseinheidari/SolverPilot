from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np

from solverpilot.capabilities import BackendManifest
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.exceptions import BackendUnavailableError


@dataclass(frozen=True, slots=True)
class BackendSolveResult:
    backend_status: str
    x: np.ndarray | None
    objective_reported: float | None = None
    raw_statistics: dict[str, object] | None = None


@runtime_checkable
class Backend(Protocol):
    @property
    def manifest(self) -> BackendManifest: ...

    def is_available(self) -> bool: ...

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult: ...
