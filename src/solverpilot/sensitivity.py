"""Local QP derivatives, conditional on a regular and stable active set."""
from dataclasses import dataclass
import numpy as np
from scipy import sparse
from solverpilot._immutability import readonly_array
from solverpilot.problem import QuadraticProblem, ObjectiveSense
from solverpilot.validate.optimality import verify_optimality


@dataclass(frozen=True, slots=True)
class QPSensitivityResult:
    status: str
    dx: np.ndarray | None
    objective_derivative: float | None
    active_rows: tuple[int,...]
    condition_estimate: float | None
    reason: str
    problem_data_hash: str
    execution_id: str | None

    def __post_init__(self):
        if self.dx is not None: object.__setattr__(self,'dx',readonly_array(self.dx,dtype=float))


def differentiate_qp(problem, result, *, dc=None, dP=None, dA=None,
                     dlower=None, dupper=None, doffset=0., active_tolerance=1e-6,
                     margin=1e-5, max_condition=1e12, max_dimension=256):
    """Differentiate P,c,A and bounds; endpoint order is linear rows then bounds.

    Returns undefined for weak activity, dependent active constraints, nonunique
    or ill-conditioned KKT systems. Equality endpoint directions must agree.
    No pseudoinverse is interpreted as a unique derivative.
    """
    if not isinstance(problem,QuadraticProblem) or problem.objective_sense is not ObjectiveSense.MINIMIZE:
        raise TypeError('sensitivity requires a continuous convex minimization QP')
    digest=getattr(getattr(result,'trace',None),'problem_data_hash',getattr(result,'problem_data_hash',None))
    if digest!=problem.data_hash: raise ValueError('result differs from the QP snapshot')
    for name,value in [('active_tolerance',active_tolerance),('margin',margin),('max_condition',max_condition)]:
        if isinstance(value,bool) or not np.isfinite(value) or value<=0: raise ValueError(f'{name} must be positive and finite')
    if type(max_dimension) is not int or max_dimension<1: raise ValueError('max_dimension must be positive')
    if isinstance(doffset,bool) or not np.isfinite(doffset): raise ValueError('doffset must be finite')
    def answer(status,reason,dx=None,derivative=None,active=(),condition=None):
        return QPSensitivityResult(status,dx,derivative,tuple(active),condition,reason,problem.data_hash,getattr(result,'execution_id',None))
    n=problem.n_variables; linear=problem.linear; m=linear.n_constraints+n
    if n>max_dimension: return answer('undefined','dense KKT dimension budget exceeded')
    def direction(value,shape,name):
        a=np.zeros(shape) if value is None else np.asarray(value.toarray() if sparse.issparse(value) else value,dtype=float)
        if a.shape!=shape or not np.isfinite(a).all(): raise ValueError(f'{name} must be finite with shape {shape}')
        return a
    cdir=direction(dc,(n,),'dc'); pdir=direction(dP,(n,n),'dP')
    if not np.array_equal(pdir,pdir.T): raise ValueError('dP must be symmetric')
    adir=None if dA is None else sparse.csr_matrix(dA,dtype=float)
    if adir is not None and (adir.shape!=linear.A.shape or not np.isfinite(adir.data).all()): raise ValueError('dA must match finite constraint coefficient directions')
    ldir=direction(dlower,(m,),'dlower'); udir=direction(dupper,(m,),'dupper')
    x=getattr(result,'x',None); raw=getattr(result,'raw_statistics',None) or {}
    y=raw.get('prepared_canonical_dual',raw.get('canonical_dual'))
    if x is None or y is None: return answer('undefined','validated primal and canonical multipliers are required')
    x=np.asarray(x,dtype=float); y=np.asarray(y,dtype=float)
    try: proof=verify_optimality(problem,x,y)
    except ValueError as exc: return answer('undefined','convex numerical KKT eligibility failed: '+str(exc))
    if not proof.verified: return answer('undefined','independent numerical KKT check failed: '+proof.reason)
    B=sparse.vstack([linear.A,sparse.eye(n)],format='csr')
    lo=np.r_[linear.constraint_lower,linear.variable_lower]; hi=np.r_[linear.constraint_upper,linear.variable_upper]
    activities=np.asarray(B@x).reshape(-1); active=[]; rhs=[]
    for i,(v,l,u) in enumerate(zip(activities,lo,hi)):
        if l==u:
            if ldir[i]!=udir[i]: raise ValueError('equality endpoint directions must agree')
            active.append(i); rhs.append(ldir[i]); continue
        lower=np.isfinite(l) and abs(v-l)<=active_tolerance; upper=np.isfinite(u) and abs(v-u)<=active_tolerance
        if lower and upper: return answer('undefined','ambiguous active bounds')
        if lower or upper:
            if (lower and y[i]>=-margin) or (upper and y[i]<=margin): return answer('undefined','weakly active inequality')
            active.append(i); rhs.append(ldir[i] if lower else udir[i])
        elif min(v-l,u-v)<margin: return answer('undefined','inactive slack margin too small')
    if n+len(active)>max_dimension: return answer('undefined','dense KKT dimension budget exceeded',active=active)
    C=B[active].toarray(); dC=np.zeros((len(active),n))
    if adir is not None:
        for local,row in enumerate(active):
            if row<linear.n_constraints: dC[local]=adir.getrow(row).toarray().reshape(-1)
    if len(active) and np.linalg.matrix_rank(C)<len(active): return answer('undefined','active constraints are linearly dependent',active=active)
    P=problem.P.toarray(); K=np.block([[P,C.T],[C,np.zeros((len(active),len(active)))]])
    condition=float(np.linalg.cond(K))
    if not np.isfinite(condition) or condition>max_condition: return answer('undefined','nonunique or ill-conditioned KKT system',active=active,condition=condition)
    right=np.r_[-cdir-pdir@x-dC.T@y[active],np.asarray(rhs)-dC@x]
    solution=np.linalg.solve(K,right); dx=solution[:n]
    residual=float(np.linalg.norm(K@solution-right,np.inf))
    if not np.isfinite(dx).all() or residual>1e-8*max(1.,float(np.linalg.norm(right,np.inf))): return answer('undefined','directional KKT residual exceeded tolerance',active=active,condition=condition)
    derivative=float(.5*x@pdir@x+cdir@x+(P@x+linear.c)@dx+doffset)
    return answer('defined','local derivative conditional on a stable active set',dx,derivative,active,condition)
