from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

import numpy as np
from scipy import sparse

from solverpilot.backends import ScipyHighsBackend
from solverpilot.problem import LinearProblem, ObjectiveSense, VariableDomain
from solverpilot.runtime import SolveResult, execute
from solverpilot.validate import CandidateSolution, validate_solution

from .errors import TSPCompileError, TSPValidationError
from .model import TSPInstance
from .solution import TSPSolution, _INDEPENDENT_PROOF_TOKEN
from .validation import TSPValidationReport, validate_tsp_route
from .distance import route_cost


@dataclass(frozen=True, slots=True)
class TSPMILPCompilation:
    problem: LinearProblem
    arc_variables: Mapping[tuple[int, int], int]
    order_variables: Mapping[int, int]
    start_index: int


@dataclass(frozen=True, slots=True)
class TSPMILPSolveResult:
    solution: TSPSolution | None
    validation: TSPValidationReport | None
    core_result: SolveResult


def _start(instance: TSPInstance, start_index: int) -> int:
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index < instance.n_nodes:
        raise TSPValidationError("start_index must be an in-range integer")
    return start_index


def compile_tsp_milp(instance: TSPInstance, *, start_index: int = 0) -> TSPMILPCompilation:
    start = _start(instance, start_index)
    n = instance.n_nodes
    arcs = {(i, j): k for k, (i, j) in enumerate(( (i,j) for i in range(n) for j in range(n) if i != j ))}
    order_nodes = tuple(i for i in range(n) if i != start)
    orders = {node: len(arcs) + k for k, node in enumerate(order_nodes)}
    nvars = len(arcs) + len(orders)
    c = np.zeros(nvars, dtype=np.float64)
    lower = np.zeros(nvars, dtype=np.float64)
    upper = np.ones(nvars, dtype=np.float64)
    domains: list[VariableDomain] = [VariableDomain.BINARY] * len(arcs) + [VariableDomain.CONTINUOUS] * len(orders)
    for (i, j), idx in arcs.items():
        c[idx] = instance.distances.values[i][j]
    for node, idx in orders.items():
        lower[idx] = 1.0
        upper[idx] = float(max(1, n - 1))
    row_i: list[int] = []
    col_i: list[int] = []
    data: list[float] = []
    cl: list[float] = []
    cu: list[float] = []
    row = 0
    for i in range(n):
        for j in range(n):
            if i != j:
                row_i.append(row); col_i.append(arcs[(i,j)]); data.append(1.0)
        cl.append(1.0); cu.append(1.0); row += 1
    for j in range(n):
        for i in range(n):
            if i != j:
                row_i.append(row); col_i.append(arcs[(i,j)]); data.append(1.0)
        cl.append(1.0); cu.append(1.0); row += 1
    # MTZ subtour elimination on non-start nodes: u_i - u_j + n*x_ij <= n-1.
    for i in order_nodes:
        for j in order_nodes:
            if i == j:
                continue
            row_i.extend((row,row,row))
            col_i.extend((orders[i],orders[j],arcs[(i,j)]))
            data.extend((1.0,-1.0,float(n)))
            cl.append(-np.inf); cu.append(float(n-1)); row += 1
    A = sparse.csr_matrix((data,(row_i,col_i)), shape=(row,nvars), dtype=np.float64)
    problem = LinearProblem.from_data(
        A=A,
        c=c,
        variable_lower=lower,
        variable_upper=upper,
        constraint_lower=np.asarray(cl,dtype=np.float64),
        constraint_upper=np.asarray(cu,dtype=np.float64),
        domains=domains,
        objective_sense=ObjectiveSense.MINIMIZE,
        name=f"tsp-mtz:{instance.name}",
        metadata={"application":"tsp","formulation":"directed-mtz","n_nodes":n,"start_index":start},
    )
    return TSPMILPCompilation(problem, MappingProxyType(dict(arcs)), MappingProxyType(dict(orders)), start)


def decode_tsp_milp_solution(instance: TSPInstance, compilation: TSPMILPCompilation, x: np.ndarray) -> tuple[int, ...]:
    values = np.asarray(x, dtype=np.float64)
    if values.ndim != 1 or values.shape[0] != compilation.problem.n_variables or not np.isfinite(values).all():
        raise TSPCompileError("MILP candidate vector has invalid shape or non-finite values")
    canonical = validate_solution(compilation.problem, CandidateSolution(x=values))
    if not canonical.valid:
        raise TSPCompileError("MILP candidate failed canonical formulation validation: " + "; ".join(canonical.warnings))
    selected = {(i,j) for (i,j), idx in compilation.arc_variables.items() if values[idx] > 0.5}
    n = instance.n_nodes
    outgoing: dict[int,int] = {}
    incoming: dict[int,int] = {}
    for i,j in selected:
        if i in outgoing or j in incoming:
            raise TSPCompileError("MILP candidate does not encode exactly one incoming/outgoing arc")
        outgoing[i]=j; incoming[j]=i
    if set(outgoing) != set(range(n)) or set(incoming) != set(range(n)):
        raise TSPCompileError("MILP candidate does not cover every TSP node")
    route=[compilation.start_index]
    cur=compilation.start_index
    for _ in range(n):
        cur=outgoing[cur]; route.append(cur)
    if cur != compilation.start_index or len(set(route[:-1])) != n:
        raise TSPCompileError("MILP candidate contains a subtour instead of one Hamiltonian cycle")
    return tuple(route)


def solve_tsp_milp(instance: TSPInstance, *, backend=None, start_index: int = 0) -> TSPMILPSolveResult:
    compilation = compile_tsp_milp(instance, start_index=start_index)
    core = execute(compilation.problem, ScipyHighsBackend() if backend is None else backend)
    if core.x is None:
        return TSPMILPSolveResult(None, None, core)
    if core.validation is None or not core.validation.valid:
        raise TSPCompileError("MILP backend returned a candidate that failed SolverPilot core validation")
    route = decode_tsp_milp_solution(instance, compilation, core.x)
    objective = route_cost(instance, route)
    solution = TSPSolution(route, objective, method="milp-mtz", is_exact=True, optimality_proven=core.optimality_evidence.independently_verified_optimal, _proof_token=_INDEPENDENT_PROOF_TOKEN, metadata={"backend_status": core.backend_status, "backend": core.trace.backend})
    report = validate_tsp_route(instance, solution)
    if not report.valid:
        raise TSPCompileError("decoded MILP solution failed independent TSP validation")
    return TSPMILPSolveResult(solution, report, core)
