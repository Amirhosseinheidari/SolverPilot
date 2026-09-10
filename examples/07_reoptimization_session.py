"""Repeated solves with explicit mutation/reuse accounting."""

import numpy as np

from solverpilot import LinearProblem, Session

problem = LinearProblem.from_data(
    A=[[1.0, 1.0]],
    c=[1.0, 2.0],
    variable_lower=[0.0, 0.0],
    variable_upper=[1.0, 1.0],
    constraint_lower=[1.0],
    constraint_upper=[np.inf],
)

session = Session(problem)
first = session.solve()
mutation = session.update(c=[2.0, 1.0])
second = session.solve()

print("first objective:", first.objective)
print("second objective:", second.objective)
print("session revision:", session.revision)
print("mutation:", mutation)
print("reuse assessment:", session.last_reuse_assessment)

assert first.validation is not None and first.validation.valid
assert second.validation is not None and second.validation.valid
assert session.revision == 1
assert not mutation.is_noop
