"""SOC semantic modeling and independent original-space validation."""
import numpy as np
from solverpilot.model import Model
from solverpilot.conic import ConicProblem, ConeKind

m = Model("soc-model")
x = m.variable(2, name="x")
t = m.variable(lower=0.0, name="t")
m.soc(t, x, name="soc")
m.minimize(t)

compiled = m.compile(use_cache=False)
problem = compiled.execution_ir

print("execution_ir:", type(problem).__name__)
print("cone_kind:", problem.cones[0].kind.value)
print("schema:", compiled.schema_version)

assert isinstance(problem, ConicProblem)
assert problem.cones[0].kind is ConeKind.SECOND_ORDER
assert compiled.validate_original(m, np.array([3.0, 4.0, 5.0]), atol=1e-8).valid
assert not compiled.validate_original(m, np.array([3.0, 4.0, 4.0]), atol=1e-8).valid
