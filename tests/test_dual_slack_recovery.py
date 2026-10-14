"""Independent exact checks for the inequality-slack recovery bound."""
from fractions import Fraction
from itertools import combinations, product

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.validate.lp_dual import dual_slack_residual_lower_bound


def exact(value):
    return Fraction(float(value))


def residual_and_base(problem, dual):
    """Construct the original Lagrangian independently of certificate helpers."""
    sign = -1 if problem.objective_sense.value == "maximize" else 1
    a = problem.A.toarray()
    residual = [sign * exact(c) + exact(dual[problem.n_constraints + j])
                + sum((exact(a[i, j]) * exact(dual[i])
                       for i in range(problem.n_constraints)), Fraction())
                for j, c in enumerate(problem.c)]
    lo = [*problem.constraint_lower, *problem.variable_lower]
    hi = [*problem.constraint_upper, *problem.variable_upper]
    base = -sum((exact(y) * exact(hi[i] if y > 0 else lo[i])
                 for i, y in enumerate(dual) if y), Fraction())
    return residual, base


def exact_vertices_2d(problem):
    """Enumerate original half-space intersections using rational arithmetic."""
    halfspaces = []
    rows = list(problem.A.toarray()) + [np.array([1., 0.]), np.array([0., 1.])]
    lo = [*problem.constraint_lower, *problem.variable_lower]
    hi = [*problem.constraint_upper, *problem.variable_upper]
    for row, lower, upper in zip(rows, lo, hi):
        a, b = map(exact, row)
        if np.isfinite(upper):
            halfspaces.append((a, b, exact(upper)))
        if np.isfinite(lower):
            halfspaces.append((-a, -b, -exact(lower)))
    vertices = set()
    for (a, b, rhs), (c, d, other_rhs) in combinations(halfspaces, 2):
        determinant = a*d-b*c
        if not determinant:
            continue
        point = ((rhs*d-b*other_rhs)/determinant,
                 (a*other_rhs-rhs*c)/determinant)
        if all(u*point[0]+v*point[1] <= bound for u, v, bound in halfspaces):
            vertices.add(point)
    assert vertices, "The independent oracle requires a nonempty bounded polygon"
    return vertices


def recovered_objective_bound(problem, dual, **kwargs):
    residual, base = residual_and_base(problem, dual)
    correction = dual_slack_residual_lower_bound(problem, residual,
        problem.variable_lower, problem.variable_upper, np.asarray(dual), **kwargs)
    return None if correction is None else base + correction


@pytest.mark.parametrize("lower,upper,c,dual,expected", [
    (-np.inf, 3., -1., 2., -3),
    (3., np.inf, 1., -2., 3),
    (-1., 2., 1., -2., -1),
    (-1., 2., -1., 2., -2),
])
def test_original_inequality_side_and_existing_slack_penalty(lower, upper, c, dual, expected):
    # The stationarity residual alone is unbounded. Its original inequality
    # slack penalty is required to recover the objective bound.
    p = LinearProblem.from_data(A=[[1.]], c=[c], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[lower], constraint_upper=[upper])
    assert recovered_objective_bound(p, [dual, 0.]) == expected


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("row_orientation", [1, -1])
def test_sparse_recovery_bounds_against_exact_polygon_enumeration(sense, row_orientation):
    # A bounded triangle, represented using either upper or lower rows. All
    # original variables are free, so recovery must use the slack equations.
    original_a = np.array([[.5, .5], [-1., 0.], [0., -1.]])
    original_rhs = np.array([1.5, 0., 0.])
    a, rhs = row_orientation*original_a, row_orientation*original_rhs
    lower = np.full(3, -np.inf) if row_orientation == 1 else rhs
    upper = rhs if row_orientation == 1 else np.full(3, np.inf)
    recovered = 0
    for cost in product([-.5, 0., .5], repeat=2):
        p = LinearProblem.from_data(A=a, c=cost, variable_lower=[-np.inf]*2,
            variable_upper=[np.inf]*2, constraint_lower=lower,
            constraint_upper=upper, objective_sense=sense)
        vertices = exact_vertices_2d(p)
        sign = -1 if sense == "maximize" else 1
        optimum = min(sum((sign*exact(c)*v for c, v in zip(cost, point)), Fraction())
                      for point in vertices)
        dual = [row_orientation*.25, row_orientation*.5, row_orientation*.75, 0., 0.]
        bound = recovered_objective_bound(p, dual)
        if bound is not None:
            assert isinstance(bound, Fraction)
            assert bound <= optimum
            recovered += 1
    assert recovered >= 5, "Exercise successful elimination as well as safe failures"


def test_variable_bound_duals_cannot_make_returned_bound_exceed_exact_optimum():
    p = LinearProblem.from_data(A=[[1., 1.], [-1., 0.], [0., -1.]], c=[-.5, .25],
        variable_lower=[0., 0.], variable_upper=[2., 2.],
        constraint_lower=[-np.inf]*3, constraint_upper=[3., 0., 0.])
    optimum = min(sum((exact(c)*v for c, v in zip(p.c, point)), Fraction())
                  for point in exact_vertices_2d(p))
    for lower_dual, upper_dual in product([.25, .5, 1.], repeat=2):
        dual = [.5, .25, .75, -lower_dual, upper_dual]
        bound = recovered_objective_bound(p, dual,
            basis={"basic_columns": [0, 1], "nonbasic_rows": [0, 1, 2]})
        assert bound is None or bound <= optimum


@pytest.mark.parametrize("tiny", [2.**-40, np.nextafter(0., 1.)])
def test_numerically_active_inequality_does_not_remove_tiny_unbounded_direction(tiny):
    p = LinearProblem.from_data(A=[[1.]], c=[tiny], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[-np.inf], constraint_upper=[0.])
    # x=0 is numerically active, but the true objective decreases toward -inf.
    assert recovered_objective_bound(p, [0., 0.], x=np.array([0.])) is None


def test_maximization_uses_signed_objective_and_preserves_tiny_exact_residual():
    p = LinearProblem.from_data(A=[[1.]], c=[2.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[-np.inf], constraint_upper=[3.],
        objective_sense="maximize")
    dual = [2. + 2.**-40, 0.]
    residual, _ = residual_and_base(p, dual)
    assert residual == [Fraction(1, 2**40)]
    assert recovered_objective_bound(p, dual) == -6


def test_variable_bound_slack_penalty_repairs_multiplier_overshoot():
    p = LinearProblem.from_data(A=np.empty((0, 1)), c=[1.], variable_lower=[0.],
        variable_upper=[np.inf], constraint_lower=[], constraint_upper=[])
    dual = [-1. - 2.**-40]
    residual, _ = residual_and_base(p, dual)
    assert residual == [-Fraction(1, 2**40)]
    assert recovered_objective_bound(p, dual) == 0


@pytest.mark.parametrize("basis", [
    {"basic_columns": [], "nonbasic_rows": []},
    {"basic_columns": [1, 0], "nonbasic_rows": []},
    {"basic_columns": [0], "nonbasic_rows": [1]},
])
def test_structurally_valid_but_wrong_basis_is_only_a_hint(basis):
    p = LinearProblem.from_data(A=[[1., 1.], [-1., 0.], [0., -1.]], c=[.5, -.5],
        variable_lower=[-np.inf]*2, variable_upper=[np.inf]*2,
        constraint_lower=[-np.inf]*3, constraint_upper=[3., 0., 0.])
    dual = [.5, .25, .75, 0., 0.]
    optimum = min(sum((exact(c)*v for c, v in zip(p.c, point)), Fraction())
                  for point in exact_vertices_2d(p))
    bound = recovered_objective_bound(p, dual, basis=basis)
    assert bound is None or bound <= optimum


@pytest.mark.parametrize("basis", [
    [], {}, {"basic_columns": []},
    {"basic_columns": [-1], "nonbasic_rows": []},
    {"basic_columns": [1], "nonbasic_rows": []},
    {"basic_columns": [0.5], "nonbasic_rows": []},
    {"basic_columns": [True], "nonbasic_rows": []},
    {"basic_columns": [0, 0], "nonbasic_rows": []},
    {"basic_columns": [], "nonbasic_rows": [-1]},
    {"basic_columns": [], "nonbasic_rows": [1]},
    {"basic_columns": [], "nonbasic_rows": [0.5]},
    {"basic_columns": [], "nonbasic_rows": [True]},
    {"basic_columns": [], "nonbasic_rows": [0, 0]},
])
def test_malformed_basis_is_rejected_before_recovery(basis):
    p = LinearProblem.from_data(A=[[1.]], c=[-1.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[-np.inf], constraint_upper=[3.])
    with pytest.raises(ValueError, match="basis"):
        recovered_objective_bound(p, [2., 0.], basis=basis)


def test_recovery_does_not_mutate_original_model_or_dual():
    p = LinearProblem.from_data(A=[[1.]], c=[-1.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[-np.inf], constraint_upper=[3.])
    original_hash = p.data_hash
    dual = np.array([2., 0.])
    original_dual = dual.copy()
    assert recovered_objective_bound(p, dual) == -3
    assert p.data_hash == original_hash
    np.testing.assert_array_equal(dual, original_dual)
