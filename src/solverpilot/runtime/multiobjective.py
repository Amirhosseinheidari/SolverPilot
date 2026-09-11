"""Sequential lexicographic linear objectives with explicit tolerance locks."""
from dataclasses import dataclass, replace
import numpy as np
from scipy import sparse
from solverpilot.problem import LinearProblem, ObjectiveSense
from solverpilot.validate import PublicStatus
from .auto import solve


@dataclass(frozen=True, slots=True)
class LexicographicResult:
    stages: tuple
    objective_values: tuple[float, ...]
    completed: bool
    tolerance: float

    @property
    def x(self):
        return None if not self.stages else self.stages[-1].x


def solve_lexicographic(problem: LinearProblem, objectives, *, senses=None, tolerance=1e-7, **solve_options):
    """Optimize linear objectives in order, including on mixed-integer models.

    Each completed stage is solver-reported optimal and independently primal
    validated. Earlier objectives may degrade by at most ``tolerance`` plus the
    configured numerical feasibility tolerance; no exact symbolic proof is claimed.
    """
    if not isinstance(problem, LinearProblem):
        raise TypeError('lexicographic objectives currently require LinearProblem')
    if not np.isfinite(tolerance) or tolerance < 0:
        raise ValueError('tolerance must be finite and nonnegative')
    objectives = tuple(np.asarray(c, dtype=float).copy() for c in objectives)
    if not objectives or any(c.shape != (problem.n_variables,) or not np.isfinite(c).all() for c in objectives):
        raise ValueError('objectives must be nonempty finite vectors matching the variables')
    senses = tuple(ObjectiveSense(s) for s in senses) if senses is not None else (ObjectiveSense.MINIMIZE,)*len(objectives)
    if len(senses) != len(objectives):
        raise ValueError('one sense is required per objective')
    current = problem; results = []
    for c, sense in zip(objectives, senses):
        current = replace(current, c=c, objective_sense=sense, objective_offset=0.)
        result = solve(current, **solve_options); results.append(result)
        if result.status is not PublicStatus.VALID_OPTIMAL or not result.validation.valid:
            return LexicographicResult(tuple(results), (), False, float(tolerance))
        value = float(c@result.x)
        lower, upper = (-np.inf, value+tolerance) if sense is ObjectiveSense.MINIMIZE else (value-tolerance, np.inf)
        current = replace(current, A=sparse.vstack([current.A, sparse.csr_matrix(c.reshape(1, -1))], format='csr'),
                          constraint_lower=np.r_[current.constraint_lower, lower], constraint_upper=np.r_[current.constraint_upper, upper])
    return LexicographicResult(tuple(results), tuple(float(c@results[-1].x) for c in objectives), True, float(tolerance))
