"""Binary MILP: maximize 3x + 2y subject to x + y <= 1."""

import numpy as np

from solverpilot import LinearProblem, VariableDomain, solve

problem = LinearProblem.from_data(
    A=[[1.0, 1.0]],
    c=[3.0, 2.0],
    variable_lower=[0.0, 0.0],
    variable_upper=[1.0, 1.0],
    constraint_lower=[-np.inf],
    constraint_upper=[1.0],
    domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    objective_sense="maximize",
)

result = solve(problem)

print("status:", result.status.value)
print("objective:", result.objective)
print("x:", result.x)
print("backend:", result.trace.backend)
print("validated:", result.validation.valid if result.validation else None)

assert result.status.value == "valid_optimal"
assert result.validation is not None and result.validation.valid
assert result.objective is not None and abs(result.objective - 3.0) < 1e-8
