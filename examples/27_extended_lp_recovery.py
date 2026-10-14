"""A tiny dual residual on an unbounded variable needs a valid domain bound."""
import numpy as np
from scipy import sparse
from solverpilot import LinearProblem
from solverpilot.validate.lp_dual import recover_lp_optimality


problem = LinearProblem.from_data(A=sparse.csr_matrix((0, 1)), c=[1.],
    variable_lower=[0.], variable_upper=[np.inf], constraint_lower=[], constraint_upper=[])
# The point is feasible. A slightly perturbed lower-bound multiplier has a tiny
# negative stationarity residual; dropping that residual would not be a proof.
check = recover_lp_optimality(problem, [0.], [-1.-2.**-40], time_limit_s=1.)
assert check.verified and check.gap == 0.
print({'verified_within_tolerances': check.verified, 'gap': check.gap,
       'recovery': check.recovery_diagnostics})
