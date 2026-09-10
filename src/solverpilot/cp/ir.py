from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Any, Mapping
import hashlib, json

class CPObjectiveSense(str, Enum):
    MINIMIZE='minimize'
    MAXIMIZE='maximize'

@dataclass(frozen=True, slots=True)
class CPIntVarIR:
    var_id:int
    name:str
    domain:tuple[int,...]
    kind:str='int'
    derived_from:tuple[int,int] | None=None  # (start_var_id, fixed_size)
    def __post_init__(self):
        if not self.domain: raise ValueError('CP variable domain cannot be empty')
        vals=tuple(sorted(set(int(v) for v in self.domain)))
        object.__setattr__(self,'domain',vals)
        if self.kind not in {'int','bool'}: raise ValueError('invalid CP variable kind')
        if self.kind=='bool' and not set(vals).issubset({0,1}): raise ValueError('bool domain must be subset of {0,1}')

@dataclass(frozen=True, slots=True)
class CPLinearExprIR:
    terms:tuple[tuple[int,int],...]=()
    constant:int=0
    def __post_init__(self):
        merged={}
        for vid,c in self.terms:
            merged[int(vid)]=merged.get(int(vid),0)+int(c)
        object.__setattr__(self,'terms',tuple(sorted((v,c) for v,c in merged.items() if c)))
        object.__setattr__(self,'constant',int(self.constant))

@dataclass(frozen=True, slots=True)
class CPIntervalIR:
    interval_id:int
    name:str
    start_var:int
    size:int
    end_var:int
    def __post_init__(self):
        if self.size < 0: raise ValueError('interval size must be nonnegative')

@dataclass(frozen=True, slots=True)
class CPLinearConstraintIR:
    expr:CPLinearExprIR; lower:int|None=None; upper:int|None=None; kind:str='linear'
@dataclass(frozen=True, slots=True)
class CPAllDifferentIR:
    var_ids:tuple[int,...]; kind:str='all_different'
@dataclass(frozen=True, slots=True)
class CPExactlyOneIR:
    literal_ids:tuple[int,...]; kind:str='exactly_one'
@dataclass(frozen=True, slots=True)
class CPTableIR:
    var_ids:tuple[int,...]; allowed_tuples:tuple[tuple[int,...],...]; kind:str='table'
@dataclass(frozen=True, slots=True)
class CPElementIR:
    index_var:int; values:tuple[int,...]; target_var:int; kind:str='element'
@dataclass(frozen=True, slots=True)
class CPCircuitIR:
    arcs:tuple[tuple[int,int,int],...]; kind:str='circuit' # tail, head, bool var
@dataclass(frozen=True, slots=True)
class CPNoOverlapIR:
    interval_ids:tuple[int,...]; kind:str='no_overlap'
@dataclass(frozen=True, slots=True)
class CPCumulativeIR:
    interval_ids:tuple[int,...]; demands:tuple[int,...]; capacity:int; kind:str='cumulative'

CPConstraintIR = CPLinearConstraintIR|CPAllDifferentIR|CPExactlyOneIR|CPTableIR|CPElementIR|CPCircuitIR|CPNoOverlapIR|CPCumulativeIR

@dataclass(frozen=True, slots=True)
class CPProblem:
    variables:tuple[CPIntVarIR,...]
    intervals:tuple[CPIntervalIR,...]
    constraints:tuple[CPConstraintIR,...]
    objective:CPLinearExprIR|None=None
    objective_sense:CPObjectiveSense=CPObjectiveSense.MINIMIZE
    name:str|None=None
    metadata:Mapping[str,Any]=field(default_factory=dict)
    schema_version:str='solverpilot.cp-ir.v1'
    def __post_init__(self):
        ids=[v.var_id for v in self.variables]
        if len(ids)!=len(set(ids)): raise ValueError('duplicate CP variable id')
        iids=[i.interval_id for i in self.intervals]
        if len(iids)!=len(set(iids)): raise ValueError('duplicate interval id')
        var_map={v.var_id:v for v in self.variables}
        interval_map={i.interval_id:i for i in self.intervals}

        def require_var(vid:int, where:str):
            if int(vid) not in var_map:
                raise ValueError(f'{where} references unknown CP variable id {vid}')

        def require_bool(vid:int, where:str):
            require_var(vid,where)
            v=var_map[int(vid)]
            if v.kind!='bool' or not set(v.domain).issubset({0,1}):
                raise ValueError(f'{where} requires Boolean literal variable id {vid}')

        def require_expr(expr:CPLinearExprIR, where:str):
            for vid,_ in expr.terms: require_var(vid,where)

        for v in self.variables:
            if v.derived_from is not None:
                require_var(v.derived_from[0],f'variable {v.var_id} derived_from')
        for i in self.intervals:
            require_var(i.start_var,f'interval {i.interval_id} start_var')
            require_var(i.end_var,f'interval {i.interval_id} end_var')
        if self.objective is not None: require_expr(self.objective,'objective')

        for c in self.constraints:
            where=f'{c.kind} constraint'
            if isinstance(c,CPLinearConstraintIR):
                require_expr(c.expr,where)
            elif isinstance(c,CPAllDifferentIR):
                for vid in c.var_ids: require_var(vid,where)
            elif isinstance(c,CPExactlyOneIR):
                if not c.literal_ids: raise ValueError('exactly_one requires at least one literal')
                for vid in c.literal_ids: require_bool(vid,where)
            elif isinstance(c,CPTableIR):
                for vid in c.var_ids: require_var(vid,where)
                if any(len(t)!=len(c.var_ids) for t in c.allowed_tuples):
                    raise ValueError('table tuple arity mismatch')
            elif isinstance(c,CPElementIR):
                require_var(c.index_var,where); require_var(c.target_var,where)
                if not c.values: raise ValueError('element values cannot be empty')
            elif isinstance(c,CPCircuitIR):
                for _,_,vid in c.arcs: require_bool(vid,where)
            elif isinstance(c,CPNoOverlapIR):
                for iid in c.interval_ids:
                    if iid not in interval_map: raise ValueError(f'{where} references unknown interval id {iid}')
            elif isinstance(c,CPCumulativeIR):
                if len(c.interval_ids)!=len(c.demands): raise ValueError('cumulative demands must align with interval_ids')
                if c.capacity < 0 or any(int(d)<0 for d in c.demands):
                    raise ValueError('cumulative demands/capacity must be nonnegative')
                for iid in c.interval_ids:
                    if iid not in interval_map: raise ValueError(f'{where} references unknown interval id {iid}')
        object.__setattr__(self,'metadata',MappingProxyType(dict(self.metadata)))
    @property
    def variable_map(self): return {v.var_id:v for v in self.variables}
    @property
    def interval_map(self): return {i.interval_id:i for i in self.intervals}
    @property
    def base_variables(self): return tuple(v for v in self.variables if v.derived_from is None)
    def canonical_dict(self):
        def expr(e): return {'terms':[list(x) for x in e.terms],'constant':e.constant}
        cs=[]
        for c in self.constraints:
            d={'kind':c.kind}
            if isinstance(c,CPLinearConstraintIR): d.update(expr=expr(c.expr),lower=c.lower,upper=c.upper)
            elif isinstance(c,CPAllDifferentIR): d['var_ids']=list(c.var_ids)
            elif isinstance(c,CPExactlyOneIR): d['literal_ids']=list(c.literal_ids)
            elif isinstance(c,CPTableIR): d.update(var_ids=list(c.var_ids),allowed_tuples=[list(t) for t in c.allowed_tuples])
            elif isinstance(c,CPElementIR): d.update(index_var=c.index_var,values=list(c.values),target_var=c.target_var)
            elif isinstance(c,CPCircuitIR): d['arcs']=[list(a) for a in c.arcs]
            elif isinstance(c,CPNoOverlapIR): d['interval_ids']=list(c.interval_ids)
            elif isinstance(c,CPCumulativeIR): d.update(interval_ids=list(c.interval_ids),demands=list(c.demands),capacity=c.capacity)
            cs.append(d)
        return {'schema_version':self.schema_version,'name':self.name,'variables':[{'id':v.var_id,'name':v.name,'domain':list(v.domain),'kind':v.kind,'derived_from':None if v.derived_from is None else list(v.derived_from)} for v in self.variables],'intervals':[{'id':i.interval_id,'name':i.name,'start_var':i.start_var,'size':i.size,'end_var':i.end_var} for i in self.intervals],'constraints':cs,'objective':None if self.objective is None else expr(self.objective),'objective_sense':self.objective_sense.value,'metadata':dict(self.metadata)}
    @property
    def structural_hash(self):
        d=self.canonical_dict();
        for v in d['variables']: v['domain']=['<domain>']
        d['metadata']={}
        return hashlib.sha256(json.dumps(d,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    @property
    def data_hash(self):
        return hashlib.sha256(json.dumps(self.canonical_dict(),sort_keys=True,separators=(',',':')).encode()).hexdigest()
