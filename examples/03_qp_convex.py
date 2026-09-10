"""Continuous convex QP with optimum x=1."""

import numpy as np

from solverpilot import QuadraticProblem, solve

problem = QuadraticProblem.from_data(
    P=[[2.0]],
    A=np.zeros((0, 1)),
    q=[-2.0],
    variable_lower=[0.0],
    variable_upper=[2.0],
    constraint_lower=[],
    constraint_upper=[],
)

result = solve(problem)

print("status:", result.status.value)
print("objective:", result.objective)
print("x:", result.x)
print("backend:", result.trace.backend)
print("validated:", result.validation.valid if result.validation else None)

assert result.validation is not None and result.validation.valid
assert result.x is not None and abs(float(result.x[0]) - 1.0) < 1e-5
assert result.objective is not None and abs(result.objective + 1.0) < 1e-7
