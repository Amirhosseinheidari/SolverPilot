"""Infeasible LP with opt-in diagnostics."""

import numpy as np

from solverpilot import LinearProblem, solve

problem = LinearProblem.from_data(
    A=[[1.0]],
    c=[1.0],
    variable_lower=[0.0],
    variable_upper=[1.0],
    constraint_lower=[2.0],
    constraint_upper=[np.inf],
)

result = solve(
    problem,
    backend="scipy-highs-ds",
    diagnose_infeasible=True,
)

print("status:", result.status.value)
print("x:", result.x)
print("validation:", result.validation)
print("diagnostics present:", result.diagnostics is not None)
if result.diagnostics is not None:
    print("diagnostics confirmed infeasible:", result.diagnostics.confirmed_infeasible)
    print("evidence order:", [item.value for item in result.diagnostics.evidence_order])

assert result.status.value == "infeasible"
assert result.x is None
assert result.validation is None
assert result.diagnostics is not None
assert result.diagnostics.confirmed_infeasible
