from __future__ import annotations
from dataclasses import dataclass
from .model import CPModel
from .reference import ReferenceCPBackend
from .ortools_backend import ORToolsCPSATBackend, VERIFIED_ORTOOLS_VERSION

@dataclass(frozen=True, slots=True)
class CPConformanceCheck:
    name:str; passed:bool; detail:str
@dataclass(frozen=True, slots=True)
class CPConformanceReport:
    backend:str; available:bool; passed:bool; checks:tuple[CPConformanceCheck,...]; verified_version:str|None=None

def reference_cp_conformance():
    checks=[]
    # Latin/permutation core.
    m=CPModel(); xs=[m.int_var(1,3,f'x{i}') for i in range(3)]; m.add_all_different(xs); m.add(xs[0]==1); m.minimize(xs[2])
    r=m.solve(ReferenceCPBackend()); checks.append(CPConformanceCheck('all-different-linear',r.optimality_proven and r.validation.valid and r.objective==2,str(r.assignment)))
    # Element/table/exactly-one.
    m=CPModel(); i=m.int_var(0,2,'i'); t=m.int_var(4,9,'t'); bs=[m.bool_var(f'b{k}') for k in range(3)]; m.add_element(i,[4,7,9],t); m.add_allowed_assignments([i,t],[(0,4),(2,9)]); m.add_exactly_one(bs); m.add(i==2); r=m.solve(); checks.append(CPConformanceCheck('element-table-exactly-one',r.validation.valid and r.assignment[t.var_id]==9,str(r.assignment)))
    # Scheduling core.
    m=CPModel(); s1=m.int_var(0,3,'s1'); s2=m.int_var(0,3,'s2'); a=m.interval_var(s1,2,'a'); b=m.interval_var(s2,2,'b'); m.add_no_overlap([a,b]); m.minimize(s1+s2); r=m.solve(); checks.append(CPConformanceCheck('interval-no-overlap',r.optimality_proven and r.validation.valid and r.objective==2,str(r.assignment)))
    return CPConformanceReport('reference-cp-enumeration',True,all(c.passed for c in checks),tuple(checks),'p9-reference-v1')

def ortools_cp_sat_conformance():
    b=ORToolsCPSATBackend();
    if not b.is_available():
        return CPConformanceReport(b.name,False,False,(CPConformanceCheck('availability',False,f'OR-Tools {VERIFIED_ORTOOLS_VERSION} unavailable in release runtime'),),VERIFIED_ORTOOLS_VERSION)
    # Same representative model against adapter.
    m=CPModel(); x=[m.int_var(1,3,f'x{i}') for i in range(3)]; m.add_all_different(x); m.add(x[0]==1); m.minimize(x[2]); r=m.solve(b)
    ok=r.optimality_proven and r.validation.valid and r.objective==2
    return CPConformanceReport(b.name,True,ok,(CPConformanceCheck('solve-and-independent-validation',ok,str(r.assignment)),),VERIFIED_ORTOOLS_VERSION)
