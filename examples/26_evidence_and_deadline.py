"""Read feasibility, optimality evidence and deadline compliance separately."""
import numpy as np
from solverpilot import LinearProblem, SolveBudget, solve
from solverpilot.runtime.unified import summarize

p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.], variable_upper=[2.],
                            constraint_lower=[1.], constraint_upper=[np.inf])
result = solve(p, backend='scipy-highs-ds', budget=SolveBudget(wall_time_s=5))
view = summarize(result)
print({'feasible': view.feasible, 'optimality': view.optimality,
       'reason': view.optimality_reason, 'within_budget': view.within_budget,
       'elapsed_s': view.elapsed_s, 'enforcement': view.deadline_enforcement})
assert view.feasible and view.optimality == 'independent_numerical_bound'
