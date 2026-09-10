from __future__ import annotations
import hashlib, json
import numpy as np
from solverpilot.model.errors import CompileError
from solverpilot.model.sets import LessThan, GreaterThan, EqualTo, Interval
from solverpilot.problem import ObjectiveSense, VariableDomain
from solverpilot.model.compiler import CompiledModel, CompilationReport, _variable_layout
from solverpilot.nlp.ir import NLPConstraintBlock, NLPProblem
from .ir import MINLPProblem, ConvexityCertificate
from .convexity import prove_curvature


def _broadcast(value,size):
    try: return np.broadcast_to(np.asarray(value,dtype=float),(size,)).astype(float,copy=True)
    except Exception as e: raise CompileError('MINLP set bound cannot broadcast') from e


def compile_minlp_model(model, *, use_cache=True):
    if model.objective is None: raise CompileError('model has no objective')
    if model.objective.sense is not ObjectiveSense.MINIMIZE: raise CompileError('P8 MINLP v1 supports minimization only')
    if model.objective.expression.shape != (): raise CompileError('P8 MINLP objective must be scalar')
    if any(c.__class__.__name__=='IndicatorConstraint' for c in model.constraints):
        raise CompileError('P8 MINLP v1 does not combine indicator bridges with nonlinear orchestration')
    offsets,n,vl,vu,domains,source_vars=_variable_layout(model)
    dom_objs=tuple(VariableDomain(d) for d in domains)
    integer=tuple(i for i,d in enumerate(dom_objs) if d in {VariableDomain.INTEGER,VariableDomain.BINARY})
    binary=tuple(i for i,d in enumerate(dom_objs) if d is VariableDomain.BINARY)
    if not integer: raise CompileError('compile_minlp_model requires at least one discrete variable')
    if integer != binary:
        raise CompileError('P8 OA v1 supports binary discrete variables only; general integer MINLP is fail-closed')
    layout={vid:(offsets[vid][0],data.shape) for vid,data in model._variables.items()}
    pvals={pid:data.value.copy() for pid,data in model._parameters.items()}
    blocks=[]; source_constraints={}; ccurv=[]
    for c in model.constraints:
        if not isinstance(c.set,(LessThan,GreaterThan,EqualTo,Interval)):
            raise CompileError(f'P8 supports scalar-set algebraic constraints only, got {type(c.set).__name__}')
        size=int(np.prod(c.function.shape)) if c.function.shape else 1
        if size != 1 and (c.function.polynomial_degree is None or c.function.polynomial_degree>1):
            raise CompileError('P8 nonlinear constraints must be scalar in v1')
        if isinstance(c.set,LessThan): lo=np.full(size,-np.inf); hi=_broadcast(c.set.upper,size)
        elif isinstance(c.set,GreaterThan): lo=_broadcast(c.set.lower,size); hi=np.full(size,np.inf)
        elif isinstance(c.set,EqualTo): lo=hi=_broadcast(c.set.value,size)
        else: lo=_broadcast(c.set.lower,size); hi=_broadcast(c.set.upper,size)
        proof=prove_curvature(c.function._node,pvals)
        nonlinear=not (c.function.polynomial_degree is not None and c.function.polynomial_degree<=1)
        if nonlinear:
            if isinstance(c.set,LessThan) and proof.curvature not in {'convex','affine'}:
                raise CompileError(f'P8 cannot certify convex upper-bounded constraint {c.entity_id.value}: {proof.reason}')
            if isinstance(c.set,GreaterThan) and proof.curvature not in {'concave','affine'}:
                raise CompileError(f'P8 cannot certify concave lower-bounded constraint {c.entity_id.value}: {proof.reason}')
            if isinstance(c.set,(EqualTo,Interval)):
                raise CompileError('P8 nonlinear equality/two-sided constraints are outside exact OA v1 scope')
        ccurv.append(proof.curvature)
        blocks.append(NLPConstraintBlock(c.entity_id.value,c.function._node,lo,hi,c.function.shape,tuple(sorted(c.function.parameter_dependencies))))
        source_constraints[c.entity_id.value]={'semantic_kind':'minlp_function_in_set','nlp_rows':size,'shape':list(c.function.shape),'curvature':proof.curvature}
    op=prove_curvature(model.objective.expression._node,pvals)
    if op.curvature not in {'convex','affine'}:
        raise CompileError(f'P8 exact OA requires certified convex objective: {op.reason}')
    relax=NLPProblem(n,vl,vu,layout,pvals,model.objective.expression._node,tuple(blocks),model.objective.sense.value,model.name,{'semantic_hash':model.semantic_hash,'compiled_by':'solverpilot-p8'})
    cert=ConvexityCertificate('solverpilot.minlp.convexity-certificate.v1',op.curvature,tuple(ccurv),'p8-affine-square-exp-v1',model.data_hash,('binary-discrete-only','nonlinear equalities rejected'))
    problem=MINLPProblem(relax,tuple(d.value for d in dom_objs),integer,binary,cert,model.semantic_hash,model.data_hash,{'global_proof_scope':'certified-convex-binary-minlp'})
    payload={'semantic_hash':model.semantic_hash,'data_hash':model.data_hash,'target':'minlp-ir-v1','policy':'p8-certified-convex-binary-oa-v1','convexity':cert.objective_curvature,'constraints':cert.constraint_curvatures}
    ch=hashlib.sha256(json.dumps(payload,sort_keys=True,separators=(',',':')).encode()).hexdigest()
    pmap={pid:{'shape':list(data.shape),'patchable':False,'parameter_compile_class':'p8_full_minlp_recertificate','plan_sensitive':True} for pid,data in model._parameters.items()}
    report=CompilationReport(target='minlp-ir-v1',semantic_analysis_s=0.0,lowering_s=0.0,total_s=0.0,n_semantic_variables=len(model._variables),n_execution_variables=n,n_semantic_constraints=len(model._constraints),n_execution_constraints=relax.n_constraints,cache_status='disabled' if not use_cache else 'p8_rebuild',changed_parameters=(),reused_execution_rows=0,recompiled_execution_rows=relax.n_constraints,objective_recompiled=True,execution_mutations=(),n_transformations=0)
    return CompiledModel(execution_ir=problem,semantic_hash=model.semantic_hash,data_hash=model.data_hash,compilation_hash=ch,parameter_map=pmap,source_map={'variables':source_vars,'constraints':source_constraints,'objective':{model.objective.entity_id.value:{'execution_objective':True,'curvature':op.curvature}}},transformation_tape=(),capability_signature='p8-certified-convex-binary-minlp-v1',mutation_plan={'mode':'p8_full_recertificate','backend_session_patch':False},plan_sensitive_dependencies=tuple(sorted(model._parameters)),precondition_certificate_hashes=(hashlib.sha256(json.dumps({'schema':cert.schema,'objective_curvature':cert.objective_curvature,'constraint_curvatures':cert.constraint_curvatures,'supported_atom_policy':cert.supported_atom_policy,'data_hash':cert.data_hash,'notes':cert.notes},sort_keys=True,default=list).encode()).hexdigest(),),compilation_report=report,reconstruction_contract={'primal':'identity_by_source_map','global_proof':'only_if_orchestrator_proved','original_space_validation':'solverpilot.nlp.validate_nlp_solution'},schema_version='solverpilot.compiled-model.p8.v1')
