"""Named parameter marginals and independent production scenarios."""

from solverpilot.applications import production_model
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.analysis import parameter_sensitivity, quality_report
from solverpilot.scenarios import scenario_sweep, scenario_statistics

model = production_model([3.0, 2.0], [[1.0, 1.0]], [4.0])
compiled = model.compile()
result = compiled.solve(backend=ScipyHighsLPBackend())
print(parameter_sensitivity(model, compiled, result, "capacity"))
print(quality_report(compiled.execution_ir, result)["summary"]["optimality"])
rows = list(scenario_sweep(model, [("base", {}), ("expanded", {"capacity": [5.0]})]))
print(scenario_statistics(rows))
assert [row.summary.objective for row in rows] == [12.0, 15.0]
