from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any
import itertools, math
from solverpilot._immutability import deep_freeze
from .ir import *
from .validate import validate_cp_solution, complete_assignment

@dataclass(frozen=True, slots=True)
class CPSolveResult:
    status:str
    assignment:Mapping[int,int]|None
    objective:int|None
    validation:object
    optimality_proven:bool
    backend:str
    raw_statistics:Mapping[str,Any]

    def __post_init__(self):
        if self.assignment is not None:
            object.__setattr__(self, 'assignment', deep_freeze(dict(self.assignment)))
        object.__setattr__(self, 'raw_statistics', deep_freeze(dict(self.raw_statistics)))

class ReferenceCPBackend:
    name='reference-cp-enumeration'
    verification_only=True
    def __init__(self,max_states=1_000_000): self.max_states=int(max_states)
    def is_available(self): return True
    def solve(self,problem:CPProblem,*,max_states=None):
        limit=self.max_states if max_states is None else int(max_states)
        base=problem.base_variables
        states=math.prod(len(v.domain) for v in base)
        if states>limit:
            return CPSolveResult('unknown_state_limit',None,None,validate_cp_solution(problem,{}),False,self.name,{'states_total':states,'states_examined':0,'limit':limit})
        best=None; best_obj=None; examined=0; feasible=0
        for values in itertools.product(*(v.domain for v in base)):
            examined+=1; a={v.var_id:int(x) for v,x in zip(base,values)}; a=complete_assignment(problem,a)
            rep=validate_cp_solution(problem,a)
            if not rep.valid: continue
            feasible+=1; obj=rep.objective_recomputed
            if problem.objective is None:
                best=a; best_obj=None; break
            if best is None or (problem.objective_sense is CPObjectiveSense.MINIMIZE and obj<best_obj) or (problem.objective_sense is CPObjectiveSense.MAXIMIZE and obj>best_obj):
                best=a.copy(); best_obj=obj
        if best is None:
            # If objective-free search stops at first feasible only when found; no feasible means full enumeration.
            return CPSolveResult('infeasible',None,None,validate_cp_solution(problem,{}),True,self.name,{'states_total':states,'states_examined':examined,'feasible_seen':feasible})
        rep=validate_cp_solution(problem,best,best_obj)
        proof = problem.objective is None or examined==states
        return CPSolveResult('optimal' if problem.objective is not None and proof else 'feasible',best,best_obj,rep,proof,self.name,{'states_total':states,'states_examined':examined,'feasible_seen':feasible})
