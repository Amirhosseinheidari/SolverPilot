from fractions import Fraction
import importlib.util
from pathlib import Path
import sys

import numpy as np
import pytest

from solverpilot.model import Model, piecewise_linear_epigraph, piecewise_linear_graph
from solverpilot.model.qcqp import try_compile_qcqp
from solverpilot.conic import ConicProblem, ConeKind


def frac(value):
    return Fraction.from_float(float(value))


def cone_projection_feasible(problem, point):
    """Independent exact projection check of emitted cone/linear maps."""
    values = [frac(x) for x in point] + [Fraction(0)] * (problem.n_variables - len(point))

    def affine(matrix, offset, row):
        a = matrix.getrow(row)
        return frac(offset[row]) + sum((frac(v) * values[i] for i, v in zip(a.indices, a.data)), Fraction(0))

    for cone in problem.cones:
        assert cone.kind is ConeKind.ROTATED_SECOND_ORDER
        v = affine(cone.F, cone.g, 1)
        tail = [affine(cone.F, cone.g, i) for i in range(2, cone.F.shape[0])]
        row = cone.F.getrow(0)
        auxiliary = [(i, a) for i, a in zip(row.indices, row.data) if i >= len(point)]
        if auxiliary:
            assert len(auxiliary) == 1 and v > 0
            i, coefficient = auxiliary[0]
            values[i] = (sum((x*x for x in tail), Fraction(0)) / (2*v)
                         - affine(cone.F, cone.g, 0)) / frac(coefficient)
    for cone in problem.cones:
        u, v = affine(cone.F, cone.g, 0), affine(cone.F, cone.g, 1)
        tail = sum((affine(cone.F, cone.g, i)**2 for i in range(2, cone.F.shape[0])), Fraction(0))
        if u < 0 or v < 0 or 2*u*v < tail:
            return False
    for i in range(len(problem.constraint_lower)):
        value = affine(problem.A, np.zeros(len(problem.constraint_lower)), i)
        if np.isfinite(problem.constraint_lower[i]) and value < frac(problem.constraint_lower[i]):
            return False
        if np.isfinite(problem.constraint_upper[i]) and value > frac(problem.constraint_upper[i]):
            return False
    return True


@pytest.mark.parametrize('kind', ['squares', 'dot', 'shifted', 'weighted', 'general', 'rational_ldl'])
def test_qcqp_exact_projection_against_original_analytic_polynomial(kind):
    m = Model(kind)
    x = m.variable(2, lower=-2., upper=2.)
    if kind == 'squares':
        expression = (x**2).sum()
        function = lambda a, b: a*a + b*b
    elif kind == 'dot':
        expression = x @ x
        function = lambda a, b: a*a + b*b
    elif kind == 'shifted':
        expression = (x[0] - .1)**2 + (x[1] + .2)**2
        function = lambda a, b: (a-frac(.1))**2 + (b+frac(.2))**2
    elif kind == 'weighted':
        expression = 3*(.1*x[0]+.2*x[1])**2 + 2*x[1]**2
        function = lambda a, b: 3*(frac(.1)*a+frac(.2)*b)**2 + 2*b*b
    elif kind == 'general':
        expression = 2*x[0]*x[0] + 2*x[0]*x[1] + 2*x[1]*x[1]
        function = lambda a, b: 2*a*a + 2*a*b + 2*b*b
    else:
        expression = 3*x[0]*x[0] + 2*x[0]*x[1] + x[1]*x[1]
        function = lambda a, b: 3*a*a + 2*a*b + b*b
    constraint = m.add(expression <= 1.)
    m.minimize(-x.sum())
    compiled = m.compile()
    assert isinstance(compiled.execution_ir, ConicProblem)
    assert compiled.source_map['constraints'][constraint.entity_id.value]['cone_blocks']
    assert compiled.reconstruction_contract['certificate'] == 'execution_ir_only_qcqp_v1'
    assert compiled.precondition_certificate_hashes
    for a, b in [(0., 0.), (.25, -.5), (1., 0.), (0., 1.), (1., 1.), (-1., -.75)]:
        assert cone_projection_feasible(compiled.execution_ir, [a, b]) == (function(frac(a), frac(b)) <= 1)


def test_qcqp_vector_constraints_concave_direction_affine_rhs_and_snapshot():
    m = Model()
    x = m.variable(2, lower=-2., upper=2.)
    weight = m.parameter(value=1.)
    m.add(-(weight*x**2) >= -1.)
    m.add(x[0]**2 <= x[1] + 1.)
    m.minimize(-x.sum())
    first = m.compile()
    assert isinstance(first.execution_ir, ConicProblem)
    assert first is m.compile()
    weight.value = 2.
    second = first.refresh(m)
    assert second.data_hash != first.data_hash and second.semantic_hash == first.semantic_hash
    assert second.compilation_hash != first.compilation_hash
    assert second.execution_ir.metadata['qcqp_source_semantic_hash'] == m.semantic_hash
    with pytest.raises(ValueError, match='snapshot'):
        first.validate_original(m, np.zeros(first.execution_ir.n_variables))
    weight.value = -1.
    assert try_compile_qcqp(m) is None
    weight.value = 0.
    assert try_compile_qcqp(m) is not None


@pytest.mark.parametrize('kind', ['indefinite', 'tiny_negative', 'zero_pivot', 'equality', 'discrete', 'nonquadratic', 'dimension_limit'])
def test_qcqp_rejects_unproved_or_unsupported_bridges(kind):
    m = Model()
    x = m.variable(33 if kind == 'dimension_limit' else 2)
    if kind == 'indefinite': expression = x[0]**2 - x[1]**2
    elif kind == 'tiny_negative': expression = x[0]**2 - np.nextafter(0., 1.) * x[1]**2
    elif kind == 'zero_pivot': expression = x[0]*x[1]
    elif kind == 'nonquadratic': expression = x[0]**4
    elif kind == 'dimension_limit': expression = (x.sum()) * x[0] + (x**2).sum()
    else: expression = (x**2).sum()
    m.add(expression == 1. if kind == 'equality' else expression <= 1.)
    if kind == 'discrete': m.binary()
    m.minimize(x.sum())
    assert try_compile_qcqp(m) is None


def test_qcqp_preserves_tiny_represented_positive_weight_without_halving():
    m = Model()
    x = m.variable()
    m.add(np.nextafter(0., 1.) * x**2 <= 1.)
    m.minimize(x)
    compiled = try_compile_qcqp(m)
    assert compiled is not None
    assert np.nextafter(0., 1.) in compiled.execution_ir.A.data


def test_qcqp_zero_constraint_quadratic_objective_metadata_and_exact_validation():
    from solverpilot.problem import QuadraticProblem
    m = Model(); x = m.variable(); m.add(0.*x**2 <= 1.); m.minimize(x**2)
    compiled = m.compile()
    assert isinstance(compiled.execution_ir, QuadraticProblem)
    for metadata in [compiled.execution_ir.metadata, compiled.execution_ir.linear.metadata]:
        assert metadata['qcqp_source_semantic_hash'] == m.semantic_hash
        assert metadata['qcqp_source_data_hash'] == m.data_hash
        assert metadata['lowered_semantic_hash'] != m.semantic_hash
    assert compiled.validate_original(m, [0.]).valid


def test_qcqp_original_validation_avoids_expanded_float_cancellation_and_fails_closed(monkeypatch):
    import solverpilot.model.qcqp as module
    m = Model(); x = m.variable(); m.add((x-1e8)**2 <= .1); m.minimize(x)
    compiled = m.compile()
    assert compiled.validate_original(m, [1e8]).valid
    report = compiled.validate_original(m, [1e8+.5])
    assert not report.valid and report.max_violation == pytest.approx(.15)
    for candidate in [[], [10**400], [np.inf], [0., 0., 0.]]:
        assert not compiled.validate_original(m, candidate).valid
    for atol in [True, np.inf, -1.]:
        with pytest.raises(ValueError): compiled.validate_original(m, [], atol=atol)
    def exhausted(self, amount=1): raise module._Unsupported('test exhausted')
    monkeypatch.setattr(module._Limits, 'tick', exhausted)
    assert not compiled.validate_original(m, [1e8]).valid


def test_qcqp_initial_point_is_explicitly_unused_and_invalid_hint_rejected():
    pytest.importorskip('clarabel')
    m = Model(); x = m.variable(); m.add(x**2 <= 1.); m.minimize(x)
    c = m.compile()
    with pytest.warns(RuntimeWarning, match='does not use x0'):
        result = c.solve(x0=[0.])
    assert result.validation.valid and result.x[0] == pytest.approx(-1., abs=1e-7)
    for hint in [[], [np.nan], [10**400]]:
        with pytest.raises(ValueError, match='QCQP x0'): c.solve(x0=hint)


def test_qcqp_work_limit_is_fail_closed(monkeypatch):
    import solverpilot.model.qcqp as module
    m = Model(); x = m.variable(2); m.add((x**2).sum() <= 1.); m.minimize(x.sum())
    def exhausted(*args, **kwargs): raise module._Unsupported('test work limit')
    monkeypatch.setattr(module._Limits, 'tick', exhausted)
    assert try_compile_qcqp(m) is None


def test_qcqp_shared_graph_term_expansion_is_charged_before_allocation():
    from solverpilot.model.qcqp import _Limits, _Poly, _Squares, _Unsupported
    limits = _Limits()
    square = _Squares(_Poly.constant(0, limits), ((Fraction(1), _Poly({(0,): Fraction(1)}, limits)),))
    with pytest.raises(_Unsupported, match='work limit'):
        for _ in range(30):
            square = square + square
    assert len(square.terms) < 100000


def test_qcqp_analytic_native_solve_and_original_validation():
    pytest.importorskip('clarabel')
    m = Model(); x = m.variable(2, lower=-2, upper=2); m.add((x**2).sum() <= 1); m.minimize(-x.sum())
    compiled = m.compile(); result = compiled.solve()
    assert result.validation.valid and compiled.validate_original(m, result.x).valid
    assert result.objective_reported == pytest.approx(-np.sqrt(2.), abs=1e-6)
    np.testing.assert_allclose(compiled.reconstruct_primal(result.x), np.ones(2)/np.sqrt(2), atol=1e-6)


@pytest.mark.parametrize('helper,points,values,input_value,expected', [
    (piecewise_linear_epigraph, [0, 1, 2], [0, 1, 4], .5, .5),
    (piecewise_linear_graph, [0, 1, 2, 3], [0, 2, -1, 1], 1., 2.),
    (piecewise_linear_graph, [0, 1, 2, 3], [0, 2, -1, 1], 1.5, .5),
    (piecewise_linear_graph, [-3, -1, 2], [-2, 3, 1], -2., .5),
    (piecewise_linear_graph, [0, 2], [-1, 3], 1., 1.),
])
def test_pwl_native_solution_matches_adjacent_interpolation(helper, points, values, input_value, expected):
    m = Model(); x = m.variable(lower=input_value, upper=input_value)
    old = m.minimize(x)
    handle = helper(m, x, points, values)
    assert m.objective is old
    m.minimize(handle.output)
    compiled = m.compile(); result = compiled.solve()
    assert result.validation.valid and result.objective == pytest.approx(expected, abs=1e-7)
    assert handle.evaluate(input_value) == pytest.approx(expected)
    assert handle.validate_original(compiled, result.x).valid
    corrupted = result.x.copy()
    output_index = compiled.source_map['variables'][handle.output.id.value]['execution_slice'][0]
    corrupted[output_index] += -.01 if handle.relation == 'epigraph' else .01
    assert not handle.validate_original(compiled, corrupted).valid


def test_pwl_epigraph_is_inequality_and_graph_prevents_nonadjacent_mixture():
    m = Model(); x = m.variable(lower=1., upper=1.)
    h = piecewise_linear_epigraph(m, x, [0, 1, 2], [0, 1, 4]); m.add(h.output == 3.); m.minimize(x)
    c = m.compile(); r = c.solve()
    assert r.validation.valid and h.validate_original(c, r.x).valid
    m = Model(); x = m.variable(lower=1., upper=1.)
    h = piecewise_linear_graph(m, x, [0, 1, 2, 3], [0, 2, -1, 1]); m.add(h.output <= 0.); m.minimize(x)
    r = m.solve()
    assert r.x is None or not r.validation.valid


@pytest.mark.parametrize('points,values', [([0, 0], [1, 2]), ([1, 0], [1, 2]),
    ([0, 1], [1]), ([0, np.inf], [1, 2]), ([0, 1], [1, np.nan]),
    ([False, True], [0, 1]), ([0, 1], ['0', '1']), ([2**53, 2**53+1], [0, 1])])
def test_pwl_invalid_data_has_no_partial_model_mutation(points, values):
    m = Model(); x = m.variable(); before = (len(m.variables), len(m.constraints))
    with pytest.raises((ValueError, TypeError)):
        piecewise_linear_graph(m, x, points, values)
    assert (len(m.variables), len(m.constraints)) == before


def test_pwl_snapshot_domains_data_ownership_and_zero_tolerance_rounding():
    m = Model(); x = m.variable(lower=.25, upper=.25); p = m.parameter(value=0.)
    data = np.array([1e16, 1e16+2])
    h = piecewise_linear_graph(m, x+p, [0., 1.], data); m.minimize(h.output)
    data[:] = 0.
    assert h.values[0] == 1e16
    with pytest.raises((ValueError, AttributeError)): h.values.shape = (1, 2)
    c = m.compile()
    candidate = np.zeros(c.execution_ir.n_variables)
    for variable, values in [(x, [.25]), (h.output, [1e16]), (h.auxiliary_variables[0], [.75, .25])]:
        a, b = c.source_map['variables'][variable.id.value]['execution_slice']
        candidate[a:b] = values
    # Exact interpolated value is 1e16+.5, not representable as binary64.
    source_check = h.validate_original(c, candidate, atol=0.)
    assert not source_check.valid and source_check.relation_violation == .5
    p.value = .25
    with pytest.raises(ValueError, match='snapshot'): h.validate_original(c, candidate)
    with pytest.raises(ValueError, match='domain'): h.evaluate(-.1)
    assert not h.validate_original(c.refresh(m), []).valid
    assert h.validate_original(c.refresh(m), [10**400]).reason == 'invalid_candidate'


def test_pwl_input_shape_owner_curvature_and_outside_domain():
    m = Model(); x = m.variable()
    foreign = Model().variable()
    for bad in [foreign, m.variable(2), x**2]:
        with pytest.raises(ValueError): piecewise_linear_graph(m, bad, [0, 1], [0, 1])
    with pytest.raises(ValueError, match='convex'):
        piecewise_linear_epigraph(m, x, [0, 1, 2], [0, 2, 1])
    m = Model(); x = m.variable(lower=3, upper=4)
    h = piecewise_linear_graph(m, x, [0, 1, 2], [0, 2, 1]); m.minimize(h.output)
    r = m.solve(); assert r.x is None or not r.validation.valid


def test_inventory_independent_oracle_and_native_complete_example():
    from solverpilot.applications.inventory import enumerate_release_reference, solve_inventory_example
    reference = enumerate_release_reference()
    assert len(reference['plans']) == 9 and reference['optimal_quantities'] == ((2, 4),)
    assert reference['optimal_cvar'] == 14.
    assert enumerate_release_reference(production_values=(0, 2, 4))['optimal_cvar'] == 13.
    report = solve_inventory_example()
    assert report['completed'] and report['reference_agrees']
    np.testing.assert_allclose(report['release_quantities'], [2, 4], atol=1e-6)
    assert report['receipt_periods'] == [1, 2] and report['lead_time_periods'] == 1
    assert report['cvar'] == pytest.approx(14., abs=1e-6)
    assert report['expected_loss'] == pytest.approx(13.5, abs=1e-6)
    np.testing.assert_allclose(report['lost_demand'], 0., atol=1e-6)


def test_inventory_secondary_reference_rejects_slack_recourse_and_bad_cost_data():
    from solverpilot.applications.inventory import enumerate_release_reference, inventory_reference_agrees, build_inventory_example
    reference = enumerate_release_reference()
    assert inventory_reference_agrees(reference, [2, 4], [14, 13])
    assert not inventory_reference_agrees(reference, [2, 4], [14, 13.65])
    for bad in [(0, 2), (0, np.nan, 5), (0, 3, 5), (1, 2, 5)]:
        for function in [enumerate_release_reference, build_inventory_example]:
            with pytest.raises(ValueError): function(production_values=bad)


def benchmark_module():
    path = Path(__file__).resolve().parents[1] / 'benchmarks/compare_lp_workflows.py'
    spec = importlib.util.spec_from_file_location('proof_comparison_next', path)
    module = importlib.util.module_from_spec(spec); sys.modules[spec.name] = module; spec.loader.exec_module(module)
    return module


def test_common_benchmark_certificate_checks_actual_candidates_and_reported_objective():
    bench = benchmark_module()
    for family in ['production', 'network']:
        for case in bench.make_sequence(family, 8, 16, 7, 4):
            assert bench.common_optimality_check(case, case.optimum_x)['verified']
            assert not bench.common_optimality_check(case, np.full(case.c.shape, np.nan))['verified']
            assert not bench.independent_check(case, case.optimum_x, case.reference_objective+1.)['passed']
    case = bench.make_sequence('production', 8, 0, 7, 2)[0]
    assert bench.independent_check(case, np.zeros_like(case.c), 0.)['primal_feasible']
    assert not bench.common_optimality_check(case, np.zeros_like(case.c))['verified']
