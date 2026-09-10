"""Trust-aware Traveling Salesperson Problem application pack.

The pack deliberately keeps exact references, heuristics, formulation compilers,
and independent validation separate. It does not replace SolverPilot's canonical
optimization runtime or backend registries.
"""

from .compile_cp import (
    MAX_CP_OBJECTIVE,
    MAX_EXACT_BINARY64_INTEGER,
    TSPCPCompilation,
    TSPCPSolveResult,
    compile_tsp_cp,
    decode_tsp_cp_solution,
    solve_tsp_cp,
)
from .compile_milp import (
    TSPMILPCompilation,
    TSPMILPSolveResult,
    compile_tsp_milp,
    decode_tsp_milp_solution,
    solve_tsp_milp,
)
from .distance import TSPDistanceAnalysis, analyze_distance_matrix, route_cost
from .errors import TSPCompileError, TSPError, TSPReferenceLimitError, TSPValidationError
from .heuristics import solve_tsp_multistart_nearest_neighbor, solve_tsp_nearest_neighbor, solve_tsp_two_opt
from .model import TSPDistanceMatrix, TSPInstance, TSPNode
from .reference import solve_tsp_brute_force, solve_tsp_held_karp
from .solution import TSPSolution
from .validation import TSPValidationReport, validate_tsp_route

__all__ = [
    "MAX_CP_OBJECTIVE", "MAX_EXACT_BINARY64_INTEGER",
    "TSPCPCompilation", "TSPCPSolveResult", "TSPCompileError", "TSPDistanceAnalysis",
    "TSPDistanceMatrix", "TSPError", "TSPInstance", "TSPMILPCompilation", "TSPMILPSolveResult",
    "TSPNode", "TSPReferenceLimitError", "TSPSolution", "TSPValidationError", "TSPValidationReport",
    "analyze_distance_matrix", "compile_tsp_cp", "compile_tsp_milp", "decode_tsp_cp_solution",
    "decode_tsp_milp_solution", "route_cost", "solve_tsp_brute_force", "solve_tsp_cp",
    "solve_tsp_held_karp", "solve_tsp_milp", "solve_tsp_multistart_nearest_neighbor",
    "solve_tsp_nearest_neighbor", "solve_tsp_two_opt", "validate_tsp_route",
]
