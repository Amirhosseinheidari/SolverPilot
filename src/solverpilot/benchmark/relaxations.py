"""Materialized continuous relaxations with independent represented-data checks."""
from dataclasses import asdict
from pathlib import Path
import hashlib
import numpy as np
from solverpilot import LinearProblem,read_mps,ObjectiveSense
from solverpilot.validate.optimality import verify_optimality


def materialize_lp_relaxation(source, destination, *, reference_time_s=30.):
    """Remove integrality, write continuous MPS, and check its actual LP bound.

    Integer reference objectives are never used as relaxation references. HiGHS
    reads/writes the MPS independently; SolverPilot crosschecks canonical data and
    verifies the numerical dual bound. This is preparation, not routing authority.
    """
    if isinstance(reference_time_s,bool) or not np.isfinite(reference_time_s) or reference_time_s<=0:
        raise ValueError('reference_time_s must be positive and finite')
    source=Path(source).resolve(); destination=Path(destination).resolve()
    if destination.exists(): raise FileExistsError('refusing to overwrite a materialized model')
    import highspy
    original=read_mps(source)
    expected=LinearProblem.from_data(A=original.A,c=original.c,variable_lower=original.variable_lower,
        variable_upper=original.variable_upper,constraint_lower=original.constraint_lower,
        constraint_upper=original.constraint_upper,objective_sense=original.objective_sense,objective_offset=original.objective_offset)
    h=highspy.Highs(); h.setOptionValue('output_flag',False)
    if h.readModel(str(source))!=highspy.HighsStatus.kOk: raise ValueError('independent MPS reader rejected source')
    native=h.getLp(); native.integrality_=[]
    if h.passModel(native)!=highspy.HighsStatus.kOk: raise ValueError('native continuous relaxation rejected')
    # Do not accept a writer that silently changes represented model coefficients.
    if h.writeModel(str(destination))!=highspy.HighsStatus.kOk: raise ValueError('continuous MPS writer failed')
    problem=read_mps(destination)
    if problem.has_integer_variables or problem.data_hash!=expected.data_hash:
        raise ValueError('materialized LP differs from the intended continuous relaxation')
    h.setOptionValue('solver','simplex'); h.setOptionValue('time_limit',float(reference_time_s))
    h.setOptionValue('primal_feasibility_tolerance',1e-9); h.setOptionValue('dual_feasibility_tolerance',1e-9)
    h.run(); solution=h.getSolution(); status=h.getModelStatus()
    x=np.asarray(solution.col_value,dtype=float)
    sign=1. if problem.objective_sense is ObjectiveSense.MINIMIZE else -1.
    dual=-sign*np.r_[solution.row_dual,solution.col_dual]
    check=verify_optimality(problem,x,dual)
    objective=float(problem.c@x+problem.objective_offset) if check.primal_valid else None
    qualified=status==highspy.HighsModelStatus.kOptimal and check.verified
    return {'schema':'solverpilot.materialized-lp-reference.v1','source_sha256':hashlib.sha256(source.read_bytes()).hexdigest(),
            'materialized_sha256':hashlib.sha256(destination.read_bytes()).hexdigest(),'source_data_hash':original.data_hash,
            'data_hash':problem.data_hash,'integrality_removed':bool(original.has_integer_variables),
            'reader_agreement':True,'reference':objective if qualified else None,'reference_verified':qualified,
            'reference_kind':'continuous_relaxation_independent_numerical_bound','reference_backend':'direct-highspy-simplex',
            'reference_backend_version':h.version(),'check':asdict(check),'production_authorized':False}
