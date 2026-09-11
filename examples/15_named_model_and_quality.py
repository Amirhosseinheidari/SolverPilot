"""A capacity-limited production model with an explicit unmet-demand penalty."""
from solverpilot.model import Model, indexed_variables, soft_constraint, named_values
from solverpilot.reporting import solution_quality

m = Model('production')
production = indexed_variables(m, ['north', 'south'], lower=0, upper=5)
unmet = soft_constraint(m, production['north']+production['south'] >= 12,
                        weight=100., name='customer-demand')
m.minimize(2*production['north']+3*production['south']+unmet.penalty)
compiled = m.compile()
result = compiled.solve()
assert result.validation.valid and abs(result.objective-225.) < 1e-6
print(named_values(m, compiled, result))
print(solution_quality(compiled.execution_ir, result)['optimality_evidence'])
