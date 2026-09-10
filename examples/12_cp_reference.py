"""Dependency-free constraint-programming reference backend."""
from solverpilot.cp import CPModel

m = CPModel("tiny-cp")
x = m.int_var(0, 2, "x")
y = m.int_var(0, 2, "y")
m.add_all_different([x, y])
m.minimize(x + y)

result = m.solve()
print("status:", result.status)
print("objective:", result.objective)
print("assignment:", result.assignment)
print("optimality_proven:", result.optimality_proven)

assert result.validation.valid
assert result.optimality_proven
assert result.objective == 1
