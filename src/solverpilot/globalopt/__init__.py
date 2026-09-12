"""Explicit, tolerance-qualified global optimization with optional SCIP.

This namespace does not change the convex QuadraticProblem or binary OA contract.
"""

from .problem import GlobalQuadraticProblem, FactorableProblem
from .expressions import absolute, maximum, minimum
from .compiler import compile_global_model
from .backend import SCIPGlobalBackend, GlobalSolveResult, solve_global
from .validation import validate_global_solution

__all__ = [
    "GlobalQuadraticProblem",
    "FactorableProblem",
    "absolute",
    "maximum",
    "minimum",
    "compile_global_model",
    "SCIPGlobalBackend",
    "GlobalSolveResult",
    "solve_global",
    "validate_global_solution",
]
