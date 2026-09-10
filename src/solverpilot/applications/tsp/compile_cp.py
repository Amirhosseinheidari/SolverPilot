from __future__ import annotations

from dataclasses import dataclass
from types import MappingProxyType
from typing import Mapping

from solverpilot.cp import CPModel, CPProblem, CPSolveResult, ReferenceCPBackend, validate_cp_solution

from .distance import route_cost
from .errors import TSPCompileError, TSPValidationError
from .model import TSPInstance
from .solution import TSPSolution, _INDEPENDENT_PROOF_TOKEN
from .validation import TSPValidationReport, validate_tsp_route

MAX_EXACT_BINARY64_INTEGER = (1 << 53) - 1
MAX_CP_OBJECTIVE = (1 << 63) - 1


@dataclass(frozen=True, slots=True)
class TSPCPCompilation:
    problem: CPProblem
    arc_variables: Mapping[tuple[int, int], int]
    start_index: int


@dataclass(frozen=True, slots=True)
class TSPCPSolveResult:
    solution: TSPSolution | None
    validation: TSPValidationReport | None
    core_result: CPSolveResult


def _start(instance: TSPInstance, start_index: int) -> int:
    if isinstance(start_index, bool) or not isinstance(start_index, int) or not 0 <= start_index < instance.n_nodes:
        raise TSPValidationError("start_index must be an in-range integer")
    return start_index


def _exact_costs(instance: TSPInstance) -> dict[tuple[int,int],int]:
    costs: dict[tuple[int,int],int] = {}
    max_cost=0
    for i in range(instance.n_nodes):
        for j in range(instance.n_nodes):
            if i==j: continue
            value=instance.distances.values[i][j]
            if not value.is_integer():
                raise TSPCompileError("CP TSP compiler requires exactly integer-valued distances; use MILP for float costs")
            integer=int(value)
            if integer > MAX_EXACT_BINARY64_INTEGER:
                raise TSPCompileError("CP TSP distance exceeds the conservative exact binary64 integer range")
            max_cost=max(max_cost,integer)
            costs[(i,j)]=integer
    if instance.n_nodes * max_cost > MAX_CP_OBJECTIVE:
        raise TSPCompileError("CP TSP objective can overflow signed int64")
    return costs


def compile_tsp_cp(instance:TSPInstance, *, start_index:int=0) -> TSPCPCompilation:
    start=_start(instance,start_index)
    costs=_exact_costs(instance)
    model=CPModel(name=f"tsp-circuit:{instance.name}")
    literals={arc:model.bool_var(name=f"x_{arc[0]}_{arc[1]}") for arc in costs}
    model.add_circuit((i,j,literals[(i,j)]) for i,j in costs)
    objective=0
    for arc,lit in literals.items():
        objective = objective + costs[arc]*lit
    model.minimize(objective)
    problem=model.compile()
    # CPModel metadata cannot currently be set through the builder; keep the canonical IR untouched.
    return TSPCPCompilation(problem, MappingProxyType({arc:lit.var_id for arc,lit in literals.items()}), start)


def decode_tsp_cp_solution(instance:TSPInstance, compilation:TSPCPCompilation, assignment:Mapping[int,int]) -> tuple[int,...]:
    for key, value in assignment.items():
        if isinstance(key, bool) or not isinstance(key, int):
            raise TSPCompileError("CP assignment keys must be integer variable ids")
        if isinstance(value, bool):
            continue
        if not isinstance(value, int):
            raise TSPCompileError("CP assignment values must be exact integers")
    canonical = validate_cp_solution(compilation.problem, assignment)
    if not canonical.valid:
        raise TSPCompileError("CP candidate failed canonical formulation validation")
    selected={(i,j) for (i,j),vid in compilation.arc_variables.items() if int(assignment.get(vid,0))==1}
    n=instance.n_nodes
    outgoing:dict[int,int]={}; incoming:dict[int,int]={}
    for i,j in selected:
        if i in outgoing or j in incoming:
            raise TSPCompileError("CP candidate does not encode exactly one incoming/outgoing arc")
        outgoing[i]=j; incoming[j]=i
    if set(outgoing)!=set(range(n)) or set(incoming)!=set(range(n)):
        raise TSPCompileError("CP candidate does not cover every TSP node")
    route=[compilation.start_index]; cur=compilation.start_index
    for _ in range(n):
        cur=outgoing[cur]; route.append(cur)
    if cur!=compilation.start_index or len(set(route[:-1]))!=n:
        raise TSPCompileError("CP candidate contains a subtour instead of one Hamiltonian cycle")
    return tuple(route)


def solve_tsp_cp(instance:TSPInstance, *, backend=None, start_index:int=0, **solve_kwargs) -> TSPCPSolveResult:
    compilation=compile_tsp_cp(instance,start_index=start_index)
    cp_backend=ReferenceCPBackend() if backend is None else backend
    core=cp_backend.solve(compilation.problem,**solve_kwargs)
    if core.assignment is None:
        return TSPCPSolveResult(None,None,core)
    if getattr(core.validation, "valid", False) is not True:
        raise TSPCompileError("CP backend returned a candidate that failed SolverPilot CP validation")
    route=decode_tsp_cp_solution(instance,compilation,core.assignment)
    objective=route_cost(instance,route)
    independent_proof = bool(core.optimality_proven and isinstance(cp_backend, ReferenceCPBackend))
    solution=TSPSolution(
        route, objective, method=f"cp-circuit:{core.backend}", is_exact=True,
        optimality_proven=independent_proof,
        _proof_token=_INDEPENDENT_PROOF_TOKEN,
        metadata={"cp_status":core.status, "backend_optimality_proven": bool(core.optimality_proven), "independent_reference_proof": independent_proof},
    )
    report=validate_tsp_route(instance,solution)
    if not report.valid:
        raise TSPCompileError("decoded CP solution failed independent TSP validation")
    return TSPCPSolveResult(solution,report,core)
