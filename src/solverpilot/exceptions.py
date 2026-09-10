"""Public exception hierarchy for SolverPilot.

M30 freezes these exception classes as the stable error surface for callers that
need to distinguish environment/setup failures from model-capability failures.
Solver termination itself remains represented by :class:`solverpilot.PublicStatus`
inside ``SolveResult`` rather than by exceptions.
"""

from __future__ import annotations


class SolverPilotError(Exception):
    """Base class for stable SolverPilot runtime/planning exceptions."""


class BackendUnavailableError(SolverPilotError, RuntimeError):
    """Raised when an explicitly requested backend cannot be used in this environment."""


class CapabilityMismatchError(SolverPilotError, ValueError):
    """Raised when an explicitly chosen backend cannot represent the problem safely."""


class UnknownBackendError(SolverPilotError, KeyError):
    """Raised when a backend identifier is not registered."""


class BudgetNotSupportedError(SolverPilotError, NotImplementedError):
    """Raised when a requested hard budget cannot be enforced safely by a backend."""


class NoCompatibleBackendError(SolverPilotError, RuntimeError):
    """Raised when planning finds no available backend satisfying all hard requirements."""

