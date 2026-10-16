"""Exact reference checks for cheaper LP bounds; no wall-time assertions."""
from fractions import Fraction
import math

import numpy as np
import pytest

from solverpilot import LinearProblem
from solverpilot.validate.lp_dual import implied_lp_box
from solverpilot.validate.optimality import verify_optimality
from solverpilot.validate.workspace import ProofPolicy


def fraction_box_reference(problem, max_visits=50000, max_passes=3):
    """Straight Fraction propagation, independent of the dyadic implementation."""
    lower, upper = problem.variable_lower.copy(), problem.variable_upper.copy()
    matrix = problem.A
    visits = 0
    for _ in range(max_passes):
        changed = False
        for i in range(matrix.shape[0]):
            start, end = matrix.indptr[i:i+2]
            if visits+end-start > max_visits:
                return lower, upper
            visits += end-start
            entries = [(j, Fraction(float(v))) for j, v in
                       zip(matrix.indices[start:end], matrix.data[start:end])]
            for rhs, is_upper in ((problem.constraint_upper[i], True),
                                  (problem.constraint_lower[i], False)):
                if not math.isfinite(rhs):
                    continue
                terms = []
                for j, coefficient in entries:
                    endpoint = (lower[j] if coefficient > 0 else upper[j]) if is_upper else (
                        upper[j] if coefficient > 0 else lower[j])
                    terms.append(None if not math.isfinite(endpoint) else coefficient*Fraction(float(endpoint)))
                missing = sum(term is None for term in terms)
                total = sum((term for term in terms if term is not None), Fraction())
                for (j, coefficient), term in zip(entries, terms):
                    if missing-(term is None):
                        continue
                    exact = (Fraction(float(rhs))-total+(term or 0))/coefficient
                    try:
                        rounded = float(exact)
                    except OverflowError:
                        continue
                    toward_upper = is_upper == (coefficient > 0)
                    if toward_upper and Fraction(rounded) < exact:
                        rounded = math.nextafter(rounded, math.inf)
                    elif not toward_upper and Fraction(rounded) > exact:
                        rounded = math.nextafter(rounded, -math.inf)
                    if toward_upper and rounded < upper[j]:
                        upper[j] = rounded; changed = True
                    elif not toward_upper and rounded > lower[j]:
                        lower[j] = rounded; changed = True
                    if lower[j] > upper[j]:
                        return problem.variable_lower.copy(), problem.variable_upper.copy()
        if not changed:
            break
    return lower, upper


@pytest.mark.parametrize('seed', range(20))
@pytest.mark.parametrize('cap', [0, 15, 50000])
def test_dyadic_propagation_is_identical_to_fraction_reference(seed, cap):
    rng = np.random.default_rng(seed)
    a = np.ldexp(rng.integers(-4, 5, (9, 6)).astype(float), rng.integers(-20, 20, (9, 6)))
    witness = rng.integers(-3, 4, (5, 6)).astype(float)
    activities = a@witness.T
    lower = np.where(rng.random(6) < .5, -np.inf, -3.)
    upper = np.where(rng.random(6) < .5, np.inf, 3.)
    p = LinearProblem.from_data(A=a, c=np.ones(6), variable_lower=lower,
        variable_upper=upper, constraint_lower=activities.min(axis=1),
        constraint_upper=activities.max(axis=1))
    actual = implied_lp_box(p, max_visits=cap)
    expected = fraction_box_reference(p, max_visits=cap)
    for left, right in zip(actual, expected):
        np.testing.assert_array_equal(left, right)
    assert np.all(witness >= actual[0]) and np.all(witness <= actual[1])


@pytest.mark.parametrize('sense', ['minimize', 'maximize'])
@pytest.mark.parametrize('side', ['lower', 'upper'])
def test_redundant_bound_multiplier_rounding_does_not_destroy_proof(sense, side):
    canonical_cost = 1. if side == 'lower' else -1.
    cost = canonical_cost if sense == 'minimize' else -canonical_cost
    p = LinearProblem.from_data(A=np.empty((0, 1)), c=[cost],
        variable_lower=[2. if side == 'lower' else -np.inf],
        variable_upper=[np.inf if side == 'lower' else 2.],
        constraint_lower=[], constraint_upper=[], objective_sense=sense, objective_offset=7.)
    dual = [-canonical_cost*(1.+2.**-40)]
    check = verify_optimality(p, [2.], dual)
    assert check.verified and check.dual_bound == cost*2.+7.
    assert not verify_optimality(p, [3. if side == 'lower' else 1.], dual).verified
    assert not verify_optimality(p, [2.], [dual[0]*2]).verified
    assert not verify_optimality(p, [2.], [-dual[0]]).verified
    assert not verify_optimality(p, [2.], dual,
        proof_policy=ProofPolicy('budgeted', max_terms=1)).verified


def test_row_only_bound_does_not_hide_true_unbounded_objective():
    p = LinearProblem.from_data(A=np.empty((0, 1)), c=[-2.**-40],
        variable_lower=[0.], variable_upper=[np.inf], constraint_lower=[], constraint_upper=[])
    check = verify_optimality(p, [0.], [0.])
    assert not check.verified and check.dual_bound is None


def test_default_equality_fallback_preserves_original_checks_and_shared_limits():
    from solverpilot.validate.lp_dual import equality_residual_lower_bound
    from solverpilot.validate.workspace import verification_scope, VerificationLimit
    p = LinearProblem.from_data(A=[[1., 1.], [1., -1.]], c=[1., 0.],
        variable_lower=[-np.inf]*2, variable_upper=[np.inf]*2,
        constraint_lower=[3., 1.], constraint_upper=[3., 1.])
    dual = [-.5+2.**-40, -.5+2.**-40, 0., 0.]
    check = verify_optimality(p, [2., 1.], dual)
    assert check.verified and check.dual_bound <= 2.
    assert check.recovery_diagnostics['mode'] == 'bounded_equality'
    assert check.recovery_diagnostics['equality']['pivots'] == 2
    assert not verify_optimality(p, [3., 0.], dual).verified
    for policy in (ProofPolicy(max_terms=1), ProofPolicy(time_limit_s=1e-15)):
        limited = verify_optimality(p, [2., 1.], dual, proof_policy=policy)
        assert not limited.verified and 'limit' in limited.reason
    with pytest.raises(VerificationLimit), verification_scope(ProofPolicy(max_terms=1)):
        equality_residual_lower_bound(p, [Fraction(1), Fraction(0)],
                                     p.variable_lower, p.variable_upper)


def test_nearly_equal_row_bounds_are_not_equations():
    # A narrow interval is still an inequality. It permits decreasing x0 and
    # increasing x1 without bound; numerical activity must not make it a proof.
    p = LinearProblem.from_data(A=[[1., 1.]], c=[2.**-40, 0.],
        variable_lower=[-np.inf]*2, variable_upper=[np.inf]*2,
        constraint_lower=[1.], constraint_upper=[math.nextafter(1., math.inf)])
    check = verify_optimality(p, [1., 0.], [0., 0., 0.])
    assert not check.verified and check.dual_bound is None
    assert check.recovery_diagnostics['equality']['reason'] == 'no_equality_pivot'


@pytest.mark.parametrize('power', [-1000, -500, 500, 1000])
def test_extreme_scale_dyadic_propagation_matches_fraction_reference(power):
    scale = math.ldexp(1., power)
    p = LinearProblem.from_data(A=[[3.*scale, -scale]], c=[1., 0.],
        variable_lower=[-np.inf, 1.], variable_upper=[np.inf, 1.],
        constraint_lower=[scale], constraint_upper=[scale])
    for actual, expected in zip(implied_lp_box(p), fraction_box_reference(p)):
        np.testing.assert_array_equal(actual, expected)


@pytest.mark.parametrize('bound_multiplier', [1., 2.])
def test_public_extended_recovery_uses_consistent_row_only_lagrangian(bound_multiplier):
    p = LinearProblem.from_data(A=[[1., 1., 0.], [1., -1., 0.]], c=[-1., 0., -1.],
        variable_lower=[-np.inf, -np.inf, 0.], variable_upper=[np.inf, np.inf, 2.],
        constraint_lower=[-np.inf]*2, constraint_upper=[3., 1.])
    # Adding the rows proves x0 <= 2; z <= 2 independently. The optimum -4
    # is attained at (2,1,2), regardless of the candidate bound multiplier.
    check = verify_optimality(p, [2., 1., 2.],
        [.5+2.**-40, .5, 0., 0., bound_multiplier], extended_recovery=True)
    assert check.recovery_diagnostics['dual_slack']['reason'] == 'bounded'
    assert check.dual_bound <= -4.
    assert check.verified == (bound_multiplier == 1.)


def test_default_equality_shared_budget_can_expire_during_pivots(monkeypatch):
    import solverpilot.validate.lp_dual as module
    from solverpilot.validate.workspace import _CURRENT
    p = LinearProblem.from_data(A=[[1., 1.], [1., -1.]], c=[1., 0.],
        variable_lower=[-np.inf]*2, variable_upper=[np.inf]*2,
        constraint_lower=[3., 1.], constraint_upper=[3., 1.])
    original = module.equality_residual_lower_bound
    entered = []
    def observed(*args, **kwargs):
        entered.append(_CURRENT.get().terms)
        return original(*args, **kwargs)
    monkeypatch.setattr(module, 'equality_residual_lower_bound', observed)
    dual = [-.5+2.**-40, -.5+2.**-40, 0., 0.]
    assert verify_optimality(p, [2., 1.], dual).verified
    # Loading the two rows consumes six terms; the first pivot exceeds this.
    checked = verify_optimality(p, [2., 1.], dual,
        proof_policy=ProofPolicy(max_terms=entered[0]+6))
    assert len(entered) == 2 and not checked.verified and 'work limit' in checked.reason


def test_default_equality_late_completion_cannot_pass(monkeypatch):
    import solverpilot.validate.lp_dual as module
    import solverpilot.validate.workspace as workspace
    p = LinearProblem.from_data(A=[[1., 1.], [1., -1.]], c=[1., 0.],
        variable_lower=[-np.inf]*2, variable_upper=[np.inf]*2,
        constraint_lower=[3., 1.], constraint_upper=[3., 1.])
    original = module.equality_residual_lower_bound
    clock = [0.]
    monkeypatch.setattr(workspace, 'perf_counter', lambda: clock[0])
    def late(*args, **kwargs):
        value = original(*args, **kwargs)
        clock[0] = 2.
        return value
    monkeypatch.setattr(module, 'equality_residual_lower_bound', late)
    checked = verify_optimality(p, [2., 1.], [-.5+2.**-40, -.5+2.**-40, 0., 0.],
                              proof_policy=ProofPolicy(time_limit_s=1.))
    assert not checked.verified and 'time limit' in checked.reason


def test_empty_rows_do_not_allocate_fraction_recovery_state(monkeypatch):
    from scipy import sparse
    import solverpilot.validate.lp_dual as module
    p = LinearProblem.from_data(A=sparse.csr_matrix((2000, 1)), c=[0.],
        variable_lower=[0.], variable_upper=[1.],
        constraint_lower=np.zeros(2000), constraint_upper=np.zeros(2000))
    calls = []
    original = module.rational
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(module, 'rational', counted)
    assert module.equality_residual_lower_bound(p, [Fraction(0)],
        p.variable_lower, p.variable_upper, max_visits=0) == 0
    assert len(calls) <= 2


def test_tiny_visit_cap_does_not_prepare_unvisited_variable_endpoints(monkeypatch):
    from scipy import sparse
    import solverpilot.validate.lp_dual as module
    n = 10000
    p = LinearProblem.from_data(A=sparse.csr_matrix(([1.], ([0], [n-1])), shape=(1, n)),
        c=np.zeros(n), variable_lower=np.zeros(n), variable_upper=np.ones(n),
        constraint_lower=[0.], constraint_upper=[1.])
    original = module._dyadic; calls = []
    def counted(value):
        calls.append(value)
        return original(value)
    monkeypatch.setattr(module, '_dyadic', counted)
    lower, upper = module.implied_lp_box(p, max_visits=0)
    assert not calls
    np.testing.assert_array_equal(lower, p.variable_lower)
    np.testing.assert_array_equal(upper, p.variable_upper)
