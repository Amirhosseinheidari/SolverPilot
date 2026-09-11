"""Explicit modeling conveniences; no hidden objective changes."""
from dataclasses import dataclass
import numpy as np
from .model import Model, PendingConstraint
from .expression import Expression


def indexed_variables(model: Model, indices, *, name='x', **kwargs):
    """Create scalar variables keyed by unique hashable application identifiers."""
    keys = tuple(indices)
    if len(set(keys)) != len(keys):
        raise ValueError('variable indices must be unique')
    return {key: model.variable(name=f'{name}[{key}]', **kwargs) for key in keys}


@dataclass(frozen=True, slots=True)
class SoftConstraint:
    slack: Expression
    penalty: Expression
    constraints: tuple


def soft_constraint(model: Model, relation: PendingConstraint, *, weight=1., name='soft'):
    """Add nonnegative slack and return its penalty for explicit objective use.

    Add ``result.penalty`` to a minimization objective (subtract for maximization).
    The caller's existing objective is never modified implicitly.
    """
    if not isinstance(relation, PendingConstraint) or relation.function._model is not model:
        raise ValueError('relation must belong to the supplied model')
    if relation.relation not in {"le", "ge", "eq"}:
        raise ValueError("unsupported relation")
    if isinstance(weight, bool) or not np.isfinite(weight) or weight <= 0:
        raise ValueError('weight must be finite and positive')
    if relation.function.polynomial_degree not in (0, 1):
        raise ValueError('soft constraints currently require affine expressions')
    slack = model.variable(relation.function.shape, lower=0., name=f'{name}.slack')
    f = relation.function
    if relation.relation == 'le':
        rows = (model.add(f <= slack, name=name),)
    elif relation.relation == 'ge':
        rows = (model.add(f >= -slack, name=name),)
    elif relation.relation == 'eq':
        rows = (model.add(f <= slack, name=f'{name}.upper'), model.add(f >= -slack, name=f'{name}.lower'))
    else:
        raise ValueError('unsupported relation')
    return SoftConstraint(slack, float(weight)*slack.sum(), rows)


def named_values(model: Model, compiled, result):
    """Return table-ready scalar records; IDs disambiguate duplicate names."""
    if result.x is None:
        return ()
    if compiled.semantic_hash != model.semantic_hash or compiled.data_hash != model.data_hash:
        raise ValueError('compiled model is stale')
    expected = getattr(compiled.execution_ir, 'data_hash', compiled.data_hash)
    actual = result.trace.problem_data_hash if hasattr(result, 'trace') else getattr(result, 'problem_data_hash', None)
    if actual != expected:
        raise ValueError("result belongs to different problem data")
    x = compiled.reconstruct_primal(result.x)
    records = []
    layout = compiled.source_map['variables']
    for variable in model.variables:
        item = layout[variable.id.value]
        start, stop = item['execution_slice']
        values = x[start:stop].reshape(variable.shape)
        for index in np.ndindex(values.shape):
            records.append({'id': variable.id.value, 'name': variable.name or variable.id.value,
                            'index': index, 'value': float(values[index])})
    return tuple(records)


def diagnose_model(model: Model, **options):
    """Map canonical LP/MILP diagnostics back to user constraint/variable names."""
    from solverpilot.diagnose import diagnose_infeasibility
    from solverpilot.problem import LinearProblem
    compiled = model.compile()
    if not isinstance(compiled.execution_ir, LinearProblem):
        raise TypeError('named conflict diagnostics currently require LP/MILP')
    report = diagnose_infeasibility(compiled.execution_ir, **options)
    row_names, variable_names = {}, {}
    for constraint in model.constraints:
        source = compiled.source_map['constraints'].get(constraint.entity_id.value, {})
        for row in source.get('execution_rows', []):
            row_names[row] = constraint.name or constraint.entity_id.value
    for variable in model.variables:
        start, stop = compiled.source_map['variables'][variable.id.value]['execution_slice']
        for i in range(start, stop):
            variable_names[i] = variable.name or variable.id.value
    def named(item):
        lookup = row_names if item.kind.value.startswith('row') or item.kind.value == 'empty_row' else variable_names
        return {'kind': item.kind.value, 'index': item.index,
                'name': lookup.get(item.index, str(item.index)),
                'amount': getattr(item, 'amount', None)}
    return {'report': report,
            'static_issues': tuple(named(x) for x in report.static_issues),
            'conflict': () if report.conflict is None else tuple(named(x) for x in report.conflict.atoms),
            'relaxation': () if report.elastic is None else tuple(named(x) for x in report.elastic.violations)}
