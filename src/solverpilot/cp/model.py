from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, Mapping
from .ir import *

@dataclass(frozen=True, slots=True, eq=False)
class CPIntVar:
    model:'CPModel'; var_id:int; name:str; domain:tuple[int,...]
    def _e(self): return CPLinearExpr(self.model,{self.var_id:1},0)
    def __add__(self,o): return self._e()+o
    def __radd__(self,o): return self._e()+o
    def __sub__(self,o): return self._e()-o
    def __rsub__(self,o): return (-self._e())+o
    def __mul__(self,o): return self._e()*o
    def __rmul__(self,o): return self._e()*o
    def __neg__(self): return -self._e()
    def __le__(self,o): return self._e()<=o
    def __ge__(self,o): return self._e()>=o
    def __eq__(self,o):
        if isinstance(o,CPIntVar) and self.model is not o.model: return False
        return self._e()==o
    __hash__=object.__hash__

@dataclass(frozen=True, slots=True, eq=False)
class CPBoolVar(CPIntVar): pass

@dataclass(frozen=True, slots=True)
class CPLinearExpr:
    model:'CPModel'; terms:Mapping[int,int]; constant:int=0
    def __post_init__(self): object.__setattr__(self,'terms',{int(k):int(v) for k,v in self.terms.items() if int(v)})
    def _coerce(self,o):
        if isinstance(o,CPIntVar):
            if o.model is not self.model: raise ValueError('cross-model CP expression')
            return o._e()
        if isinstance(o,CPLinearExpr):
            if o.model is not self.model: raise ValueError('cross-model CP expression')
            return o
        if isinstance(o,int): return CPLinearExpr(self.model,{},o)
        raise TypeError(f'unsupported CP expression value {type(o).__name__}')
    def __add__(self,o):
        r=self._coerce(o); d=dict(self.terms)
        for k,v in r.terms.items(): d[k]=d.get(k,0)+v
        return CPLinearExpr(self.model,d,self.constant+r.constant)
    __radd__=__add__
    def __neg__(self): return CPLinearExpr(self.model,{k:-v for k,v in self.terms.items()},-self.constant)
    def __sub__(self,o): return self+(-self._coerce(o))
    def __rsub__(self,o): return self._coerce(o)+(-self)
    def __mul__(self,o):
        if not isinstance(o,int): raise TypeError('P9 CP linear expressions only support integer scaling')
        return CPLinearExpr(self.model,{k:o*v for k,v in self.terms.items()},o*self.constant)
    __rmul__=__mul__
    def __le__(self,o): return CPBoundedLinear(self-self._coerce(o),None,0)
    def __ge__(self,o): return CPBoundedLinear(self-self._coerce(o),0,None)
    def __eq__(self,o): return CPBoundedLinear(self-self._coerce(o),0,0)
    def ir(self): return CPLinearExprIR(tuple(self.terms.items()),self.constant)

@dataclass(frozen=True, slots=True)
class CPBoundedLinear:
    expr:CPLinearExpr; lower:int|None; upper:int|None

@dataclass(frozen=True, slots=True)
class CPIntervalVar:
    model:'CPModel'; interval_id:int; name:str; start:CPIntVar; size:int; end:CPIntVar

class CPModel:
    def __init__(self,name=None):
        self.name=name; self._vars=[]; self._intervals=[]; self._constraints=[]; self._objective=None; self._sense=CPObjectiveSense.MINIMIZE
    def int_var(self,lower:int,upper:int,name=None):
        if int(lower)>int(upper): raise ValueError('lower > upper')
        return self.int_var_from_values(range(int(lower),int(upper)+1),name)
    def int_var_from_values(self,values:Iterable[int],name=None):
        vals=tuple(sorted(set(int(x) for x in values)))
        if not vals: raise ValueError('domain cannot be empty')
        vid=len(self._vars); v=CPIntVar(self,vid,name or f'x{vid}',vals); self._vars.append((v,None,'int')); return v
    def bool_var(self,name=None):
        vid=len(self._vars); v=CPBoolVar(self,vid,name or f'b{vid}',(0,1)); self._vars.append((v,None,'bool')); return v
    def interval_var(self,start:CPIntVar,size:int,name=None):
        self._check(start); size=int(size)
        if size<0: raise ValueError('interval size must be nonnegative')
        end_dom=tuple(v+size for v in start.domain); vid=len(self._vars); end=CPIntVar(self,vid,(name or f'i{len(self._intervals)}')+'_end',end_dom); self._vars.append((end,(start.var_id,size),'int'))
        iid=len(self._intervals); iv=CPIntervalVar(self,iid,name or f'i{iid}',start,size,end); self._intervals.append(iv); return iv
    def _check(self,*vars):
        for v in vars:
            if not isinstance(v,CPIntVar) or v.model is not self: raise ValueError('CP variable belongs to another model')
    def add(self,bounded:CPBoundedLinear):
        if not isinstance(bounded,CPBoundedLinear) or bounded.expr.model is not self: raise TypeError('CPModel.add expects a bounded CP linear expression')
        self._constraints.append(CPLinearConstraintIR(bounded.expr.ir(),bounded.lower,bounded.upper)); return self
    def add_all_different(self,vars):
        vs=tuple(vars); self._check(*vs); self._constraints.append(CPAllDifferentIR(tuple(v.var_id for v in vs))); return self
    def add_exactly_one(self,literals):
        vs=tuple(literals); self._check(*vs)
        if not vs: raise ValueError('exactly_one requires literals')
        if any(tuple(v.domain)!=(0,1) for v in vs): raise ValueError('exactly_one requires BoolVar domains')
        self._constraints.append(CPExactlyOneIR(tuple(v.var_id for v in vs))); return self
    def add_allowed_assignments(self,vars,tuples_list):
        vs=tuple(vars); self._check(*vs)
        if not vs: raise ValueError('table requires variables')
        rows=tuple(tuple(int(x) for x in row) for row in tuples_list)
        if any(len(r)!=len(vs) for r in rows): raise ValueError('table tuple arity mismatch')
        self._constraints.append(CPTableIR(tuple(v.var_id for v in vs),rows)); return self
    def add_element(self,index:CPIntVar,values,target:CPIntVar):
        self._check(index,target); vals=tuple(int(v) for v in values)
        if not vals: raise ValueError('element values cannot be empty')
        self._constraints.append(CPElementIR(index.var_id,vals,target.var_id)); return self
    def add_circuit(self,arcs):
        out=[]
        for tail,head,lit in arcs:
            self._check(lit)
            if tuple(lit.domain)!=(0,1): raise ValueError('circuit arc literal must be BoolVar')
            out.append((int(tail),int(head),lit.var_id))
        if not out: raise ValueError('circuit requires arcs')
        if any(t<0 or h<0 for t,h,_ in out): raise ValueError('circuit node ids must be nonnegative')
        loops=[(t,h) for t,h,_ in out if t==h]
        if len(loops)!=len(set(loops)): raise ValueError('multiple self-loops for one node are not CP-SAT compatible')
        self._constraints.append(CPCircuitIR(tuple(out))); return self
    def add_no_overlap(self,intervals):
        ivs=tuple(intervals)
        if any(not isinstance(i,CPIntervalVar) or i.model is not self for i in ivs): raise ValueError('interval belongs to another model')
        self._constraints.append(CPNoOverlapIR(tuple(i.interval_id for i in ivs))); return self
    def add_cumulative(self,intervals,demands,capacity):
        ivs=tuple(intervals); ds=tuple(int(x) for x in demands); cap=int(capacity)
        if len(ivs)!=len(ds): raise ValueError('cumulative interval/demand length mismatch')
        if any(d<0 for d in ds): raise ValueError('cumulative demands must be nonnegative')
        if any(not isinstance(i,CPIntervalVar) or i.model is not self for i in ivs): raise ValueError('interval belongs to another model')
        self._constraints.append(CPCumulativeIR(tuple(i.interval_id for i in ivs),ds,cap)); return self
    def minimize(self,expr): self._objective=self._as_expr(expr).ir(); self._sense=CPObjectiveSense.MINIMIZE; return self
    def maximize(self,expr): self._objective=self._as_expr(expr).ir(); self._sense=CPObjectiveSense.MAXIMIZE; return self
    def _as_expr(self,x):
        if isinstance(x,CPIntVar): self._check(x); return x._e()
        if isinstance(x,CPLinearExpr):
            if x.model is not self: raise ValueError('cross-model objective')
            return x
        if isinstance(x,int): return CPLinearExpr(self,{},x)
        raise TypeError('objective must be CP linear expression')
    def compile(self):
        variables=tuple(CPIntVarIR(v.var_id,v.name,v.domain,kind,derived) for v,derived,kind in self._vars)
        intervals=tuple(CPIntervalIR(i.interval_id,i.name,i.start.var_id,i.size,i.end.var_id) for i in self._intervals)
        return CPProblem(variables,intervals,tuple(self._constraints),self._objective,self._sense,self.name,{'compiled_by':'solverpilot-p9-cp-core'})
    def solve(self,backend=None,**kwargs):
        if backend is None:
            from .reference import ReferenceCPBackend
            backend=ReferenceCPBackend()
        return backend.solve(self.compile(),**kwargs)
