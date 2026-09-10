from __future__ import annotations
from dataclasses import dataclass
import numpy as np
from solverpilot.model.expression import ExprNode
from solverpilot.model.types import Curvature

_AFFINE_KINDS={'constant','parameter','variable','index','transpose','sum','concat','matmul'}

@dataclass(frozen=True, slots=True)
class CurvatureProof:
    curvature: str
    reason: str


def _fixed_scalar(node: ExprNode, parameter_values: dict[str,np.ndarray]):
    if node.variable_dependencies:
        return None
    k=node.kind
    if k=='constant':
        a=np.asarray(node.payload,dtype=float)
    elif k=='parameter':
        a=np.asarray(parameter_values[str(node.payload)],dtype=float)
    elif k=='neg':
        v=_fixed_scalar(node.args[0],parameter_values); return None if v is None else -v
    elif k in {'add','mul','div'}:
        a0=_fixed_scalar(node.args[0],parameter_values); a1=_fixed_scalar(node.args[1],parameter_values)
        if a0 is None or a1 is None: return None
        if k=='add': return a0+a1
        if k=='mul': return a0*a1
        if np.any(np.asarray(a1)==0): return None
        return a0/a1
    elif k=='pow':
        a0=_fixed_scalar(node.args[0],parameter_values)
        return None if a0 is None else np.power(a0,int(node.payload))
    else:
        return None
    if np.asarray(a).size != 1: return None
    return float(np.asarray(a).reshape(-1)[0])


def _same_node(a: ExprNode,b: ExprNode) -> bool:
    return a is b or a == b


def prove_curvature(node: ExprNode, parameter_values: dict[str,np.ndarray]) -> CurvatureProof:
    if node.degree is not None and node.degree <= 1:
        return CurvatureProof('affine','polynomial degree <= 1')
    k=node.kind
    if k=='neg':
        p=prove_curvature(node.args[0],parameter_values)
        return CurvatureProof({'convex':'concave','concave':'convex'}.get(p.curvature,p.curvature),f'neg({p.reason})')
    if k=='add':
        a=prove_curvature(node.args[0],parameter_values); b=prove_curvature(node.args[1],parameter_values)
        if a.curvature=='affine': return CurvatureProof(b.curvature,f'affine + {b.reason}')
        if b.curvature=='affine': return CurvatureProof(a.curvature,f'{a.reason} + affine')
        if a.curvature==b.curvature and a.curvature in {'convex','concave'}:
            return CurvatureProof(a.curvature,f'sum of {a.curvature} terms')
        return CurvatureProof('unknown','sum composition not certified')
    if k=='mul':
        a,b=node.args
        ca=_fixed_scalar(a,parameter_values); cb=_fixed_scalar(b,parameter_values)
        if ca is not None:
            p=prove_curvature(b,parameter_values)
            if ca==0: return CurvatureProof('affine','zero scaling')
            if ca>0: return CurvatureProof(p.curvature,f'positive scaling {ca:g}')
            return CurvatureProof({'convex':'concave','concave':'convex'}.get(p.curvature,p.curvature),f'negative scaling {ca:g}')
        if cb is not None:
            p=prove_curvature(a,parameter_values)
            if cb==0: return CurvatureProof('affine','zero scaling')
            if cb>0: return CurvatureProof(p.curvature,f'positive scaling {cb:g}')
            return CurvatureProof({'convex':'concave','concave':'convex'}.get(p.curvature,p.curvature),f'negative scaling {cb:g}')
        pa=prove_curvature(a,parameter_values)
        if _same_node(a,b) and pa.curvature=='affine' and a.shape==():
            return CurvatureProof('convex','square of scalar affine expression')
        return CurvatureProof('unknown','bilinear/product term not certified')
    if k=='exp':
        p=prove_curvature(node.args[0],parameter_values)
        if p.curvature=='affine': return CurvatureProof('convex','exp of affine')
        return CurvatureProof('unknown','exp composition outside P8 certificate policy')
    if k=='pow':
        p=prove_curvature(node.args[0],parameter_values); e=int(node.payload)
        if p.curvature=='affine' and e>=2 and e%2==0 and node.shape==():
            return CurvatureProof('convex',f'even power {e} of scalar affine')
        return CurvatureProof('unknown','power outside P8 certificate policy')
    return CurvatureProof('unknown',f'atom {k!r} outside P8 convexity certificate policy')
