"""Compare automatic capability-aware routing with an explicit backend request."""

import numpy as np

from solverpilot import LinearProblem, solve

problem = LinearProblem.from_data(
    A=[[1.0, 1.0]],
    c=[1.0, 2.0],
    variable_lower=[0.0, 0.0],
    variable_upper=[1.0, 1.0],
    constraint_lower=[1.0],
    constraint_upper=[np.inf],
)

auto = solve(problem)
explicit = solve(problem, backend="scipy-highs-ipm")

print("automatic backend:", auto.trace.backend)
print("planner backend:", auto.plan.selected_backend if auto.plan else None)
print("explicit backend:", explicit.trace.backend)

assert auto.validation is not None and auto.validation.valid
assert auto.plan is not None
assert explicit.validation is not None and explicit.validation.valid
assert explicit.plan is None
assert explicit.trace.backend == "scipy-highs-ipm"
