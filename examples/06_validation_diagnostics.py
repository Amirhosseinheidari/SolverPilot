"""Inspect the public SolveResult and ValidationReport contract."""

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

result = solve(problem)
validation = result.validation

print("public status:", result.status.value)
print("backend status:", result.backend_status)
print("objective:", result.objective)
print("x:", result.x)
print("executed backend:", result.trace.backend)
print("planned backend:", result.plan.selected_backend if result.plan else None)

assert validation is not None
print("valid:", validation.valid)
print("max bound violation:", validation.max_bound_violation)
print("max constraint violation:", validation.max_constraint_violation)
print("max integrality violation:", validation.max_integrality_violation)
print("objective recomputed:", validation.objective_recomputed)
print("objective difference:", validation.objective_difference)
print("warnings:", validation.warnings)

assert result.status.value == "valid_optimal"
assert validation.valid
assert validation.max_bound_violation <= 1e-7
assert validation.max_constraint_violation <= 1e-7
