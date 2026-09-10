from __future__ import annotations
from dataclasses import replace
import numpy as np
from solverpilot.model.errors import CompileError
from solverpilot.model.sets import LessThan, GreaterThan, EqualTo, Interval
from solverpilot.problem import ObjectiveSense, VariableDomain
from solverpilot.model.compiler import CompiledModel, CompilationReport, _variable_layout
from .ir import NLPConstraintBlock, NLPProblem


def _broadcast_bound(value, size):
    arr=np.asarray(value,dtype=float)
    try: return np.broadcast_to(arr,(size,)).astype(float,copy=True)
    except Exception as e: raise CompileError('NLP set bound cannot broadcast to expression size') from e


def _node_interval(node, *, variable_layout, variable_lower, variable_upper, parameter_values):
    """Conservative scalar interval over all elements of an expression node.

    ``None`` means the compiler cannot prove a safe range from declared variable
    bounds and frozen parameter values. The analysis is intentionally fail-closed.
    """
    kind=node.kind
    if kind=='constant':
        a=np.asarray(node.payload,dtype=float); return (float(np.min(a)),float(np.max(a)))
    if kind=='parameter':
        a=np.asarray(parameter_values[str(node.payload)],dtype=float); return (float(np.min(a)),float(np.max(a)))
    if kind=='variable':
        off,shape=variable_layout[str(node.payload)]; size=int(np.prod(shape)) if shape else 1
        return (float(np.min(variable_lower[off:off+size])),float(np.max(variable_upper[off:off+size])))
    if kind in {'index','transpose'}:
        return _node_interval(node.args[0],variable_layout=variable_layout,variable_lower=variable_lower,variable_upper=variable_upper,parameter_values=parameter_values)
    if kind=='sum':
        child=_node_interval(node.args[0],variable_layout=variable_layout,variable_lower=variable_lower,variable_upper=variable_upper,parameter_values=parameter_values)
        if child is None: return None
        count=int(np.prod(node.args[0].shape)) if node.payload is None else node.args[0].shape[int(node.payload)]
        lo,hi=child
        if lo>=0 or hi<=0: return (lo*count,hi*count)
        return None
    vals=[_node_interval(a,variable_layout=variable_layout,variable_lower=variable_lower,variable_upper=variable_upper,parameter_values=parameter_values) for a in node.args]
    if any(v is None for v in vals): return None
    if kind=='neg': return (-vals[0][1],-vals[0][0])
    if kind=='add': return (vals[0][0]+vals[1][0],vals[0][1]+vals[1][1])
    if kind=='mul':
        a,b=vals; c=[a[0]*b[0],a[0]*b[1],a[1]*b[0],a[1]*b[1]]; return (min(c),max(c))
    if kind=='div':
        a,b=vals
        if b[0] <= 0 <= b[1]: return None
        c=[a[0]/b[0],a[0]/b[1],a[1]/b[0],a[1]/b[1]]; return (min(c),max(c))
    if kind=='pow':
        exponent=int(node.payload); a=vals[0]
        if exponent%2==0:
            hi=max(abs(a[0]),abs(a[1]))**exponent
            lo=0.0 if a[0] <= 0 <= a[1] else min(abs(a[0]),abs(a[1]))**exponent
            return (lo,hi)
        return (a[0]**exponent,a[1]**exponent)
    if kind=='exp': return (float(np.exp(vals[0][0])),float(np.exp(vals[0][1])))
    if kind=='log':
        if vals[0][0] <= 0: return None
        return (float(np.log(vals[0][0])),float(np.log(vals[0][1])))
    if kind=='sqrt':
        if vals[0][0] < 0: return None
        return (float(np.sqrt(vals[0][0])),float(np.sqrt(vals[0][1])))
    if kind=='tanh': return (float(np.tanh(vals[0][0])),float(np.tanh(vals[0][1])))
    return None

def _collect_domain_hazards(node, *, variable_layout, variable_lower, variable_upper, parameter_values, where):
    hazards=[]
    for child in node.args:
        hazards.extend(_collect_domain_hazards(child,variable_layout=variable_layout,variable_lower=variable_lower,variable_upper=variable_upper,parameter_values=parameter_values,where=where))
    if node.kind not in {'log','sqrt','div'}:
        return hazards
    target=node.args[0] if node.kind in {'log','sqrt'} else node.args[1]
    interval=_node_interval(target,variable_layout=variable_layout,variable_lower=variable_lower,variable_upper=variable_upper,parameter_values=parameter_values)
    safe=False
    if interval is not None:
        lo,hi=interval
        safe=(node.kind=='log' and lo>0) or (node.kind=='sqrt' and lo>=0) or (node.kind=='div' and not (lo<=0<=hi))
    if not safe:
        hazards.append({'where':where,'kind':node.kind,'interval':None if interval is None else [float(interval[0]),float(interval[1])],'status':'not_proven_safe_from_declared_bounds'})
    return hazards

def compile_nlp_model(model, *, use_cache=True):
    if model.objective is None: raise CompileError('model has no objective')
    if model.objective.sense is not ObjectiveSense.MINIMIZE: raise CompileError('P7 NLP compiler currently supports minimization only')
    if model.objective.expression.shape != (): raise CompileError('P7 NLP objective must be scalar')
    if any(d.domain is not VariableDomain.CONTINUOUS for d in model._variables.values()): raise CompileError('P7 NLP supports continuous variables only; MINLP is P8')
    offsets,n,vl,vu,domains,source_vars=_variable_layout(model)
    layout={vid:(offsets[vid][0],data.shape) for vid,data in model._variables.items()}
    pvals={pid:data.value.copy() for pid,data in model._parameters.items()}
    domain_hazards=_collect_domain_hazards(
        model.objective.expression._node, variable_layout=layout, variable_lower=vl, variable_upper=vu,
        parameter_values=pvals, where='NLP objective'
    )
    blocks=[]; source_constraints={}
    for c in model.constraints:
        if not hasattr(c,'set') or c.__class__.__name__=='IndicatorConstraint': raise CompileError('P7 NLP compiler does not lower indicator constraints; compile through P4/P8 orchestration')
        if not isinstance(c.set,(LessThan,GreaterThan,EqualTo,Interval)): raise CompileError(f'P7 NLP supports scalar-set constraints only, got {type(c.set).__name__}')
        domain_hazards.extend(_collect_domain_hazards(
            c.function._node, variable_layout=layout, variable_lower=vl, variable_upper=vu,
            parameter_values=pvals, where=f'NLP constraint {c.entity_id.value}'
        ))
        size=int(np.prod(c.function.shape)) if c.function.shape else 1
        if isinstance(c.set,LessThan): lo=np.full(size,-np.inf); hi=_broadcast_bound(c.set.upper,size)
        elif isinstance(c.set,GreaterThan): lo=_broadcast_bound(c.set.lower,size); hi=np.full(size,np.inf)
        elif isinstance(c.set,EqualTo): lo=hi=_broadcast_bound(c.set.value,size)
        else: lo=_broadcast_bound(c.set.lower,size); hi=_broadcast_bound(c.set.upper,size)
        blocks.append(NLPConstraintBlock(c.entity_id.value,c.function._node,lo,hi,c.function.shape,tuple(sorted(c.function.parameter_dependencies))))
        source_constraints[c.entity_id.value]={'semantic_kind':'nonlinear_function_in_set','nlp_rows':size,'shape':list(c.function.shape)}
    problem=NLPProblem(n,vl,vu,layout,pvals,model.objective.expression._node,tuple(blocks),model.objective.sense.value,model.name,{'semantic_hash':model.semantic_hash,'compiled_by':'solverpilot-p7','domain_hazards':tuple(domain_hazards),'domain_safety_proven':not bool(domain_hazards)})
    import hashlib,json
    payload={'semantic_hash':model.semantic_hash,'target':'nlp-ir-v1','policy':'p7-smooth-core-v1','objective_degree':model.objective.expression.polynomial_degree,'constraint_kinds':[c.function._node.kind for c in model.constraints]}
    ch=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pmap={pid:{'shape':list(data.shape),'patchable':False,'parameter_compile_class':'p7_full_nlp_relower','plan_sensitive':bool(pid in model.objective.expression.parameter_dependencies or any(pid in c.function.parameter_dependencies for c in model.constraints))} for pid,data in model._parameters.items()}
    report=CompilationReport(target='nlp-ir-v1',semantic_analysis_s=0.0,lowering_s=0.0,total_s=0.0,n_semantic_variables=len(model._variables),n_execution_variables=n,n_semantic_constraints=len(model._constraints),n_execution_constraints=problem.n_constraints,cache_status='disabled' if not use_cache else 'p7_rebuild',changed_parameters=(),reused_execution_rows=0,recompiled_execution_rows=problem.n_constraints,objective_recompiled=True,execution_mutations=(),n_transformations=0)
    return CompiledModel(execution_ir=problem,semantic_hash=model.semantic_hash,data_hash=model.data_hash,compilation_hash=ch,parameter_map=pmap,source_map={'variables':source_vars,'constraints':source_constraints,'objective':{model.objective.entity_id.value:{'execution_objective':True}}},transformation_tape=(),capability_signature='p7-nlp-smooth-core-casadi-ad-v1',mutation_plan={'mode':'p7_full_nlp_relower','backend_session_patch':False},plan_sensitive_dependencies=tuple(sorted(set().union(*[set(c.function.parameter_dependencies) for c in model.constraints],set(model.objective.expression.parameter_dependencies)))),precondition_certificate_hashes=(),compilation_report=report,reconstruction_contract={'primal':'identity_by_source_map','dual':'backend_specific','original_space_validation':'solverpilot.nlp.validate_nlp_solution'},schema_version='solverpilot.compiled-model.p7.v1')
