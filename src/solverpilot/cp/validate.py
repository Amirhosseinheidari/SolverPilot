from __future__ import annotations
from dataclasses import dataclass
from typing import Mapping
import math
from .ir import *

@dataclass(frozen=True, slots=True)
class CPValidationReport:
    valid:bool
    domain_violations:tuple[int,...]
    constraint_violations:tuple[int,...]
    derived_violations:tuple[int,...]
    objective_recomputed:int|None
    objective_reported:int|None
    objective_consistent:bool|None

def _expr(e, a): return e.constant + sum(c*a[v] for v,c in e.terms)

def complete_assignment(problem,assignment):
    a={int(k):int(v) for k,v in dict(assignment).items()}
    pending=True
    while pending:
        pending=False
        for v in problem.variables:
            if v.var_id in a: continue
            if v.derived_from is not None:
                s,size=v.derived_from
                if s in a: a[v.var_id]=a[s]+size; pending=True
    return a

def _circuit_ok(c,a):
    nodes=set(); selected=[]
    for t,h,l in c.arcs:
        nodes.add(t); nodes.add(h)
        if a[l]==1: selected.append((t,h))
    # CP-SAT semantics: incident nodes exactly one selected incoming/outgoing.
    for n in nodes:
        if sum(1 for t,h in selected if t==n)!=1: return False
        if sum(1 for t,h in selected if h==n)!=1: return False
    nonself=[e for e in selected if e[0]!=e[1]]
    if not nonself: return True
    active=set(x for e in nonself for x in e)
    start=next(iter(active)); seen={start}; cur=start
    for _ in range(len(active)):
        nxts=[h for t,h in nonself if t==cur]
        if len(nxts)!=1: return False
        cur=nxts[0]
        if cur==start: break
        if cur in seen: return False
        seen.add(cur)
    return cur==start and seen==active

def validate_cp_solution(problem:CPProblem,assignment:Mapping[int,int],objective_reported=None):
    a=complete_assignment(problem,assignment); dviol=[]; derived=[]; cviol=[]
    for v in problem.variables:
        if v.var_id not in a or a[v.var_id] not in v.domain: dviol.append(v.var_id)
        if v.derived_from is not None and v.var_id in a:
            s,size=v.derived_from
            if s not in a or a[v.var_id] != a[s]+size: derived.append(v.var_id)
    imap=problem.interval_map
    for idx,c in enumerate(problem.constraints):
        ok=True
        try:
            if isinstance(c,CPLinearConstraintIR):
                val=_expr(c.expr,a); ok=(c.lower is None or val>=c.lower) and (c.upper is None or val<=c.upper)
            elif isinstance(c,CPAllDifferentIR): ok=len({a[v] for v in c.var_ids})==len(c.var_ids)
            elif isinstance(c,CPExactlyOneIR): ok=sum(a[v] for v in c.literal_ids)==1
            elif isinstance(c,CPTableIR): ok=tuple(a[v] for v in c.var_ids) in set(c.allowed_tuples)
            elif isinstance(c,CPElementIR):
                ix=a[c.index_var]; ok=0<=ix<len(c.values) and a[c.target_var]==c.values[ix]
            elif isinstance(c,CPCircuitIR): ok=_circuit_ok(c,a)
            elif isinstance(c,CPNoOverlapIR):
                spans=[(a[imap[i].start_var],a[imap[i].end_var]) for i in c.interval_ids]
                ok=all(e1<=s2 or e2<=s1 for j,(s1,e1) in enumerate(spans) for s2,e2 in spans[j+1:])
            elif isinstance(c,CPCumulativeIR):
                items=[(a[imap[i].start_var],a[imap[i].end_var],d) for i,d in zip(c.interval_ids,c.demands)]
                points=sorted({s for s,e,d in items if s<e})
                ok=c.capacity>=0 and all(sum(d for s,e,d in items if s<=t<e)<=c.capacity for t in points)
        except KeyError: ok=False
        if not ok: cviol.append(idx)
    obj=None if problem.objective is None or any(v not in a for v,c in problem.objective.terms) else _expr(problem.objective,a)
    consistent=None if objective_reported is None or obj is None else int(objective_reported)==int(obj)
    return CPValidationReport(not dviol and not derived and not cviol and consistent is not False,tuple(dviol),tuple(cviol),tuple(derived),obj,None if objective_reported is None else int(objective_reported),consistent)
