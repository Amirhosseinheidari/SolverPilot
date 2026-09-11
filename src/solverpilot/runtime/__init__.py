from .auto import builtin_backend_candidates, default_registry, solve, solve_production
from .budgeting import apply_budget
from .executor import execute
from .portfolio import PortfolioAttempt, PortfolioSolveResult, execute_portfolio
from .result import OptimalityEvidence, SolveResult
from .options import SolveOptions, ProgressEvent
from .unified import SolutionSummary, summarize, solve_any, solve_verified, UnverifiedSolutionError

__all__ = [
    "PortfolioAttempt",
    "PortfolioSolveResult",
    "SolveResult",
    "OptimalityEvidence",
    "apply_budget",
    "builtin_backend_candidates",
    "default_registry",
    "execute",
    "execute_portfolio",
    "solve",
    "solve_production",
    "SolveOptions", "ProgressEvent", "SolutionSummary", "summarize", "solve_any",
    "solve_verified", "UnverifiedSolutionError",
]
