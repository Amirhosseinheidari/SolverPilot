from __future__ import annotations


class TSPError(ValueError):
    """Base error for the SolverPilot TSP application pack."""


class TSPValidationError(TSPError):
    """Raised when a TSP model, route, or solution is invalid."""


class TSPCompileError(TSPError):
    """Raised when an exact compiler cannot preserve TSP semantics safely."""


class TSPReferenceLimitError(TSPError):
    """Raised when an exponential exact reference solver hits its safety guard."""
