"""Analytic parameter-to-data directions for unbridged polynomial LP/QP models."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from solverpilot._immutability import readonly_array
from solverpilot.problem import LinearProblem,QuadraticProblem
from solverpilot.problem.linear import _readonly_csr
from .compiler import _Polynomial,_variable_layout,_rows_to_csr
from .model import Parameter
from .sets import LessThan,GreaterThan,EqualTo,Interval


@dataclass(frozen=True, slots=True)
class ParameterDataDirection:
    problem_data_hash: str
    model_data_hash: str
    parameter_id: str
    parameter_direction: np.ndarray
    dc: np.ndarray
    dP: sparse.csr_matrix
    dA: sparse.csr_matrix
    dlower: np.ndarray
    dupper: np.ndarray
    doffset: float

    def __post_init__(self):
        for name in ('parameter_direction','dc','dlower','dupper'):
            value=getattr(self,name)
            if not np.isfinite(value).all(): raise ValueError('nonfinite parameter data direction')
            object.__setattr__(self,name,readonly_array(value,dtype=float))
        for name in ('dP','dA'):
            value=getattr(self,name)
            if not np.isfinite(value.data).all(): raise ValueError('nonfinite parameter matrix direction')
            object.__setattr__(self,name,_readonly_csr(value))
        if not np.isfinite(self.doffset): raise ValueError('nonfinite objective offset direction')


@dataclass
class _Jet:
    value: _Polynomial
    tangent: _Polynomial

    @staticmethod
    def constant(value,tangent=0.): return _Jet(_Polynomial.scalar(value),_Polynomial.scalar(tangent))
    @staticmethod
    def coerce(value): return value if isinstance(value,_Jet) else _Jet.constant(value)
    def __add__(self,other):
        other=self.coerce(other); return _Jet(self.value+other.value,self.tangent+other.tangent)
    __radd__=__add__
    def __neg__(self): return _Jet(-self.value,-self.tangent)
    def __mul__(self,other):
        other=self.coerce(other)
        return _Jet(self.value*other.value,self.tangent*other.value+self.value*other.tangent)
    __rmul__=__mul__


def parameter_data_direction(model, parameter, direction=None, *, max_expression_elements=100000):
    """Differentiate the polynomial lowering algebra, without parameter perturbation.

    Supports the current LP/QP parameter DAG: affine constraints, quadratic
    objectives, indexing, broadcasting, sums, transpose, concat and matmul.
    Bridges/convex atoms/nonlinear paths raise explicitly. No model mutation occurs.
    """
    if type(max_expression_elements) is not int or max_expression_elements<1: raise ValueError('expression element cap must be positive')
    if isinstance(parameter,str):
        matches=[p for p in model.parameters if p.name==parameter]
        if len(matches)!=1: raise ValueError('parameter name must identify exactly one parameter')
        parameter=matches[0]
    if not isinstance(parameter,Parameter) or parameter._model is not model: raise ValueError('parameter must belong to this model')
    compiled=model.compile(); problem=compiled.execution_ir
    if not isinstance(problem,(LinearProblem,QuadraticProblem)) or compiled.schema_version!='solverpilot.compiled-model.p2.v1' or compiled.transformation_tape:
        raise NotImplementedError('parameter data directions require an unbridged polynomial LP/QP')
    tangent=np.ones(parameter.shape) if direction is None else np.broadcast_to(np.asarray(direction,dtype=float),parameter.shape).copy()
    if not np.isfinite(tangent).all(): raise ValueError('parameter direction must be finite')
    offsets,n,*_= _variable_layout(model); cache={}; visited_elements=0
    def evaluate(node):
        nonlocal visited_elements
        key=id(node)
        if key in cache: return cache[key]
        visited_elements+=int(np.prod(node.shape)) if node.shape else 1
        if visited_elements>max_expression_elements: raise ValueError('parameter lowering element budget exceeded')
        kind=node.kind
        if kind in {'constant','parameter','variable'}:
            out=np.empty(node.shape,dtype=object)
            if kind=='constant': raw=node.payload.toarray() if sparse.issparse(node.payload) else np.asarray(node.payload)
            elif kind=='parameter': raw=model._parameters[node.payload].value
            for local,index in enumerate(np.ndindex(node.shape)):
                if kind=='variable': out[index]=_Jet(_Polynomial.variable(offsets[node.payload][0]+local),_Polynomial.scalar(0))
                else: out[index]=_Jet.constant(float(raw[index]),float(tangent[index]) if kind=='parameter' and node.payload==parameter.id.value else 0.)
        else:
            args=[evaluate(a) for a in node.args]
            if kind=='add': out=args[0]+args[1]
            elif kind=='mul': out=args[0]*args[1]
            elif kind=='neg': out=-args[0]
            elif kind=='matmul': out=np.asarray(args[0]@args[1],dtype=object)
            elif kind=='index': out=np.asarray(args[0][node.payload],dtype=object)
            elif kind=='transpose': out=args[0].T
            elif kind=='sum': out=np.asarray(np.sum(args[0],axis=node.payload),dtype=object).reshape(node.shape)
            elif kind=='concat': out=np.concatenate([a.reshape(-1) for a in args])
            else: raise NotImplementedError(f'parameter direction for {kind} is unsupported')
        out=np.asarray(out,dtype=object).reshape(node.shape)
        cache[key]=out; return out
    objective=_Jet.coerce(evaluate(model.objective.expression._node).item()).tangent
    c=np.zeros(n)
    for i,value in objective.linear.items(): c[i]=value
    rr=[]; cc=[]; values=[]
    for (i,j),value in objective.quadratic.items():
        rr.append(i); cc.append(j); values.append(2*value if i==j else value)
        if i!=j: rr.append(j); cc.append(i); values.append(value)
    P=sparse.csr_matrix((values,(rr,cc)),shape=(n,n)); rows=[]; lower=[]; upper=[]
    for constraint in model.constraints:
        for item in evaluate(constraint.function._node).reshape(-1):
            poly=_Jet.coerce(item).tangent
            if poly.quadratic: raise NotImplementedError('quadratic constraint directions are unsupported')
            rows.append(poly.linear)
            lower.append(-poly.constant if isinstance(constraint.set,(GreaterThan,EqualTo,Interval)) else 0.)
            upper.append(-poly.constant if isinstance(constraint.set,(LessThan,EqualTo,Interval)) else 0.)
    return ParameterDataDirection(problem.data_hash,model.data_hash,parameter.id.value,tangent,c,P,_rows_to_csr(rows,n),
                                  np.r_[lower,np.zeros(n)],np.r_[upper,np.zeros(n)],float(objective.constant))


def differentiate_parameter_qp(model, result, parameter, direction=None, **regularity_controls):
    """Chain analytic semantic parameter directions through regular local QP KKT."""
    from solverpilot.sensitivity import differentiate_qp
    data=parameter_data_direction(model,parameter,direction)
    return differentiate_qp(model.compile().execution_ir,result,dc=data.dc,dP=data.dP,dA=data.dA,
                            dlower=data.dlower,dupper=data.dupper,doffset=data.doffset,**regularity_controls)
