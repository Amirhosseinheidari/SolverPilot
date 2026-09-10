"""Semantic Model -> canonical LP -> validated solve."""
from solverpilot.model import Model

m = Model("semantic-lp")
x = m.variable(2, lower=0.0, upper=5.0, name="x")
cost = m.parameter(2, value=[2.0, 1.0], name="cost")
m.add(x[0] + x[1] >= 1.0)
m.minimize(cost @ x)

compiled = m.compile()
result = compiled.solve()

print("execution_ir:", type(compiled.execution_ir).__name__)
print("status:", result.status.value)
print("objective:", result.objective)
print("x:", result.x)
print("cache_status:", compiled.compilation_report.cache_status)

assert type(compiled.execution_ir).__name__ == "LinearProblem"
assert result.validation is not None and result.validation.valid
assert abs(result.objective - 1.0) < 1e-8

# Parameter-only update: semantic structure stays fixed, data changes.
semantic_hash = m.semantic_hash
cost.value = [1.0, 3.0]
updated = m.compile()
assert updated.semantic_hash == semantic_hash
assert updated.data_hash != compiled.data_hash
