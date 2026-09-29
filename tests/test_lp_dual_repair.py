from dataclasses import replace
from fractions import Fraction

import numpy as np
import pytest

from solverpilot import LinearProblem, solve
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.backends.base import BackendSolveResult
from solverpilot.validate.lp_dual import implied_lp_box, prepare_lp_dual, row_residual_lower_bound
from solverpilot.validate.optimality import verify_optimality


def free_box():
    return LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[1.], constraint_upper=[2.])


def test_exact_row_bounds_recover_finite_residual_correction():
    p = free_box()
    check = verify_optimality(p, [1.], [-1.+1e-12, 0.])
    assert check.verified and check.domain_refined
    assert check.dual_bound <= 1.


def test_truly_unbounded_residual_stays_unverified():
    p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
        variable_upper=[np.inf], constraint_lower=[1.], constraint_upper=[np.inf])
    check = verify_optimality(p, [1.], [-1.-1e-12, 0.])
    assert not check.verified and check.dual_bound is None


def test_coupled_free_variables_use_row_form_not_false_finite_boxes():
    p = LinearProblem.from_data(A=[[1., -1.]], c=[1., -1.], variable_lower=[-np.inf]*2,
        variable_upper=[np.inf]*2, constraint_lower=[1.], constraint_upper=[1.])
    assert np.isposinf(implied_lp_box(p)[1]).all()
    check = verify_optimality(p, [1., 0.], [-1.+1e-12, 0., 0.])
    assert check.verified and check.domain_refined and check.dual_bound <= 1.
    lo, hi = p.variable_lower, p.variable_upper
    assert row_residual_lower_bound(p, [Fraction(1), Fraction(-1)], lo, hi) == 1
    assert row_residual_lower_bound(p, [Fraction(1), Fraction(0)], lo, hi) is None
    assert row_residual_lower_bound(p, [Fraction(1), Fraction(-1)], lo, hi, max_visits=0) is None


def test_wrong_inequality_direction_cannot_bound_a_residual():
    p = LinearProblem.from_data(A=[[1., -1.]], c=[1., -1.], variable_lower=[-np.inf]*2,
        variable_upper=[np.inf]*2, constraint_lower=[1.], constraint_upper=[np.inf])
    assert row_residual_lower_bound(p, [Fraction(1), Fraction(-1)], p.variable_lower, p.variable_upper) == 1
    assert row_residual_lower_bound(p, [Fraction(-1), Fraction(1)], p.variable_lower, p.variable_upper) is None


def test_projected_dual_is_not_automatically_a_proof():
    p = free_box()
    raw = [-1., 1e-12]
    fixed = prepare_lp_dual(p, raw)
    assert raw == [-1., 1e-12] and fixed.tolist() == [-1., 0.]
    assert verify_optimality(p, [1.], fixed).verified
    assert not verify_optimality(p, [2.], fixed).verified
    assert prepare_lp_dual(p, [1.]) is None
    assert prepare_lp_dual(p, [np.nan, 0.]) is None
    assert prepare_lp_dual(p, ['bad', 0.]) is None
    assert prepare_lp_dual(object(), raw) is None


def test_outward_rounding_preserves_exact_fractional_endpoint():
    p = LinearProblem.from_data(A=[[3.]], c=[1.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[1.], constraint_upper=[1.])
    lo, hi = implied_lp_box(p)
    assert Fraction(float(lo[0])) <= Fraction(1, 3) <= Fraction(float(hi[0]))
    assert lo[0] < hi[0]


def test_propagation_and_work_cap_are_sound():
    p = LinearProblem.from_data(A=[[1., -1.], [0., 3.]], c=[1., 0.],
        variable_lower=[-np.inf, -np.inf], variable_upper=[np.inf, np.inf],
        constraint_lower=[0., 3.], constraint_upper=[0., 6.])
    lo, hi = implied_lp_box(p)
    np.testing.assert_equal(lo, [1., 1.]); np.testing.assert_equal(hi, [2., 2.])
    lo, hi = implied_lp_box(p, max_visits=0)
    assert np.isneginf(lo).all() and np.isposinf(hi).all()


def test_exact_cancellation_and_contradictions_do_not_fabricate_bounds():
    p = LinearProblem.from_data(A=[[1., 1e16, 1., -1e16]], c=[1., 0., 0., 0.],
        variable_lower=[-np.inf, 1., 1., 1.], variable_upper=[np.inf, 1., 1., 1.],
        constraint_lower=[-np.inf], constraint_upper=[2.])
    assert implied_lp_box(p)[1][0] == 1.
    p = LinearProblem.from_data(A=[[1.], [1.]], c=[1.], variable_lower=[-np.inf],
        variable_upper=[np.inf], constraint_lower=[2., -np.inf], constraint_upper=[np.inf, 1.])
    lo, hi = implied_lp_box(p)
    assert np.isneginf(lo[0]) and np.isposinf(hi[0])


@pytest.mark.parametrize('seed', range(5))
def test_derived_box_contains_independent_feasible_witnesses(seed):
    rng = np.random.default_rng(seed)
    points = rng.integers(-4, 5, size=(20, 6)).astype(float)
    a = rng.integers(-5, 6, size=(15, 6)).astype(float)
    values = a @ points.T
    p = LinearProblem.from_data(A=a, c=np.ones(6), variable_lower=np.full(6, -4.),
        variable_upper=np.full(6, np.inf), constraint_lower=values.min(axis=1),
        constraint_upper=values.max(axis=1))
    lo, hi = implied_lp_box(p)
    assert np.all(points >= lo) and np.all(points <= hi)


def test_executor_keeps_raw_dual_and_checks_prepared_candidate():
    class NoisyBackend:
        manifest = replace(ScipyHighsLPBackend().manifest, name='noisy-proof-test')
        def is_available(self): return True
        def solve(self, problem):
            return BackendSolveResult('optimal', np.array([1.]), 1.,
                                      {'canonical_dual': [-1., 1e-12]})
    result = solve(free_box(), backend=NoisyBackend())
    assert result.optimality_evidence.independently_verified_optimal
    assert result.raw_statistics['canonical_dual'][1] == 1e-12
    assert result.raw_statistics['prepared_canonical_dual'][1] == 0.
    assert not result.raw_statistics['original_optimality_check']['verified']
