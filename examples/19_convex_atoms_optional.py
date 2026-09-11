"""Convex robust loss with an SOC constraint; requires Clarabel."""

from importlib.util import find_spec
import numpy as np
from solverpilot.model import Model, norm, huber, named_values

if find_spec("clarabel") is None:
    print("Install solverpilot[clarabel] to run this convex-atom example.")
else:
    model = Model()
    x = model.variable(2, lower=-10.0, upper=10.0, name="fit")
    model.add(norm(x) <= 1.0)
    model.minimize(huber(x - model.constant([3.0, 4.0])).sum())
    compiled = model.compile()
    result = compiled.solve()
    assert result.validation.valid
    assert np.linalg.norm(compiled.reconstruct_primal(result.x)) <= 1.0 + 1e-6
    print(named_values(model, compiled, result))
