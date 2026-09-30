"""Original-objective numerical bounds checked against exact analytic optima."""

from fractions import Fraction

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem
from solverpilot.validate import verify_optimality


def box_problem(c, lower, upper, *, sense="minimize", offset=0.):
    return LinearProblem.from_data(
        A=np.empty((0, len(c))), c=c, variable_lower=lower, variable_upper=upper,
        constraint_lower=[], constraint_upper=[], objective_sense=sense,
        objective_offset=offset,
    )


def assert_outward_nearest(bound, exact, sense):
    """No stronger than the exact bound, and no weaker than one rounding step."""
    assert bound is not None and np.isfinite(bound)
    represented = Fraction(float(bound))
    if sense == "minimize":
        assert represented <= exact
        assert Fraction(float(np.nextafter(bound, np.inf))) > exact
    else:
        assert represented >= exact
        assert Fraction(float(np.nextafter(bound, -np.inf))) < exact


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("offset", [-100., 0., 5.])
def test_box_optimum_bound_restores_objective_sense_and_offset(sense, offset):
    p = box_problem([1.], [0.], [1.], sense=sense, offset=offset)
    x, dual = ([0.], [-1.]) if sense == "minimize" else ([1.], [1.])
    check = verify_optimality(p, x, dual)
    assert check.verified and check.primal_valid and check.dual_valid
    assert check.gap == 0.
    assert check.dual_bound == offset + x[0]


@pytest.mark.parametrize("offset", [-10., 0., 10.])
def test_convex_qp_bound_includes_offset_without_changing_gap(offset):
    # x^2 - 2x + offset has its minimum at x=1 with value offset-1.
    p = QuadraticProblem.from_data(
        P=[[2.]], A=np.empty((0, 1)), q=[-2.], variable_lower=[0.],
        variable_upper=[2.], constraint_lower=[], constraint_upper=[],
        objective_offset=offset,
    )
    check = verify_optimality(p, [1.], [0.])
    assert check.verified and check.gap == 0.
    assert check.dual_bound == offset - 1.


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("coefficient", [-2.**-54, 2.**-54])
@pytest.mark.parametrize("offset", [-1., 1.])
def test_offset_conversion_rounds_outward_from_exact_binary64_data(sense, coefficient, offset):
    p = box_problem([coefficient], [1.], [1.], sense=sense, offset=offset)
    sign = 1. if sense == "minimize" else -1.
    check = verify_optimality(p, [1.], [-sign * coefficient])
    exact = Fraction(coefficient) + Fraction(offset)
    assert check.verified and check.gap == 0.
    assert_outward_nearest(check.dual_bound, exact, sense)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("coefficient_sign", [-1., 1.])
def test_subnormal_objective_product_rounds_outward(sense, coefficient_sign):
    coefficient = coefficient_sign * np.nextafter(0., 1.)
    p = box_problem([coefficient], [.5], [.5], sense=sense)
    sign = 1. if sense == "minimize" else -1.
    check = verify_optimality(p, [.5], [-sign * coefficient])
    exact = Fraction(float(coefficient)) / 2
    assert check.verified and check.gap == 0.
    assert_outward_nearest(check.dual_bound, exact, sense)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("value_sign", [-1., 1.])
def test_objective_offset_overflow_never_overstates_bound(sense, value_sign):
    largest = np.finfo(float).max
    coefficient, offset = value_sign * (largest / 2), value_sign * largest
    p = box_problem([coefficient], [1.], [1.], sense=sense, offset=offset)
    sign = 1. if sense == "minimize" else -1.
    with np.errstate(over="ignore", invalid="ignore"):
        check = verify_optimality(p, [1.], [-sign * coefficient])
    assert not check.verified  # Original primal objective is non-finite.
    assert not np.isnan(check.dual_bound)
    outward_overflow = (sense == "minimize" and value_sign < 0) or (
        sense == "maximize" and value_sign > 0
    )
    if outward_overflow:
        assert check.dual_bound == value_sign * np.inf
    else:
        assert check.dual_bound == value_sign * largest
        exact = Fraction(float(coefficient)) + Fraction(float(offset))
        if sense == "minimize":
            assert Fraction(check.dual_bound) < exact
        else:
            assert Fraction(check.dual_bound) > exact


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
@pytest.mark.parametrize("value_sign", [-1., 1.])
def test_offset_restores_finite_bound_before_float_conversion(sense, value_sign):
    largest = np.finfo(float).max
    coefficient, offset = value_sign * (largest * .75), -value_sign * largest
    p = box_problem([coefficient], [2.], [2.], sense=sense, offset=offset)
    sign = 1. if sense == "minimize" else -1.
    with np.errstate(over="ignore", invalid="ignore"):
        check = verify_optimality(p, [2.], [-sign * coefficient])
    assert not check.verified  # The numerical objective calculation overflows.
    exact = 2 * Fraction(float(coefficient)) + Fraction(float(offset))
    assert_outward_nearest(check.dual_bound, exact, sense)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_unverified_candidate_keeps_valid_original_objective_bound(sense):
    p = box_problem([1.], [0.], [1.], sense=sense, offset=-100.)
    x, dual = ([1.], [-1.]) if sense == "minimize" else ([0.], [1.])
    check = verify_optimality(p, x, dual)
    assert not check.verified and check.primal_valid
    assert check.gap == 1.
    assert check.dual_bound == (-100. if sense == "minimize" else -99.)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_missing_bound_remains_none_despite_objective_offset(sense):
    p = box_problem([1.], [-np.inf], [np.inf], sense=sense, offset=10.)
    check = verify_optimality(p, [0.], [0.])
    assert not check.verified and check.dual_bound is None
    assert np.isinf(check.gap)


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_bounds_match_exact_multivariable_analytic_controls(sense):
    c = np.array([-3., 2., .5, -.25])
    lower, upper = np.array([-2., 1., 0., -4.]), np.array([1., 3., 2., -1.])
    sign = 1. if sense == "minimize" else -1.
    x = np.where(sign * c >= 0, lower, upper)
    for offset in [-7.5, 0., 19.25]:
        p = box_problem(c, lower, upper, sense=sense, offset=offset)
        check = verify_optimality(p, x, -sign * c)
        exact = sum((Fraction(float(a)) * Fraction(float(b)) for a, b in zip(c, x)), Fraction(offset))
        assert check.verified and check.gap == 0.
        assert check.dual_bound == exact
