"""Table-ready original-coordinate solution diagnostics."""
from dataclasses import asdict
import numpy as np
from solverpilot.problem import QuadraticProblem


def solution_quality(problem, result, *, row_names=None, variable_names=None):
    """Explain primal violations and distinguish backend claims from verification."""
    linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
    if result.trace.problem_data_hash != problem.data_hash:
        raise ValueError('result belongs to different problem data')
    row_names = tuple(row_names) if row_names is not None else tuple(f'row[{i}]' for i in range(linear.n_constraints))
    variable_names = tuple(variable_names) if variable_names is not None else tuple(f'x[{i}]' for i in range(linear.n_variables))
    if len(row_names) != linear.n_constraints or len(variable_names) != linear.n_variables:
        raise ValueError('names must match problem dimensions')
    rows = []
    if result.x is not None:
        for kind, names, values, lo, hi in [
            ('constraint', row_names, linear.A@result.x, linear.constraint_lower, linear.constraint_upper),
            ('variable', variable_names, result.x, linear.variable_lower, linear.variable_upper),
        ]:
            for i, (name, value, lower, upper) in enumerate(zip(names, values, lo, hi)):
                rows.append({'kind': kind, 'index': i, 'name': str(name), 'value': float(value),
                             'lower': float(lower), 'upper': float(upper),
                             'violation': float(max(0., lower-value, value-upper)) if np.isfinite(value) else float('inf')})
    rows.sort(key=lambda r: r['violation'], reverse=True)
    return {'status': result.status.value, 'backend_status': result.backend_status,
            'objective': result.objective, 'primal_validated': bool(result.validation and result.validation.valid),
            'validation': None if result.validation is None else asdict(result.validation),
            'optimality_evidence': asdict(result.optimality_evidence),
            'timings': asdict(result.trace.timings), 'rows': rows,
            'warnings': list(result.trace.warnings)}
