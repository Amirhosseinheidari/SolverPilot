from fractions import Fraction
import numpy as np
import pytest
from solverpilot import LinearProblem
from solverpilot.validate.lp_dual import equality_residual_lower_bound, recover_lp_optimality


def coupled():
    return LinearProblem.from_data(A=[[1.,1.],[1.,-1.]], c=[1.,0.], variable_lower=[-np.inf]*2,
        variable_upper=[np.inf]*2, constraint_lower=[3.,1.],constraint_upper=[3.,1.])


def test_combined_equalities_recover_bound_on_free_variables():
    p=coupled();r=[Fraction(1),Fraction(0)]
    assert equality_residual_lower_bound(p,r,p.variable_lower,p.variable_upper)==2
    assert recover_lp_optimality(p,[2.,1.],[-.5+1e-12,-.5,0.,0.]).verified
    assert not recover_lp_optimality(p,[3.,0.],[-.5,-.5,0.,0.]).verified


def test_resource_caps_fail_closed():
    p=coupled();r=[Fraction(1),Fraction(0)]
    for options in ({'max_pivots':0},{'max_visits':0},{'max_bits':1}):
        assert equality_residual_lower_bound(p,r,p.variable_lower,p.variable_upper,**options) is None
    with pytest.raises(ValueError):recover_lp_optimality(p,[2.,1.],[np.nan]*4)


def test_inequalities_are_never_promoted_to_equalities():
    p=LinearProblem.from_data(A=[[1.]],c=[1.],variable_lower=[-np.inf],variable_upper=[np.inf],
        constraint_lower=[-np.inf],constraint_upper=[1.])
    assert equality_residual_lower_bound(p,[Fraction(1)],p.variable_lower,p.variable_upper) is None


def test_elimination_fill_in_supplies_missing_original_column_pivot():
    p = LinearProblem.from_data(A=[[1., 1., 0.], [1., 0., 1.]], c=[1., 0., 0.],
        variable_lower=[-np.inf, -np.inf, 0.], variable_upper=[np.inf, np.inf, 1.],
        constraint_lower=[3., 4.], constraint_upper=[3., 4.])
    details = {}
    bound = equality_residual_lower_bound(p, [Fraction(1), Fraction(0), Fraction(0)],
        p.variable_lower, p.variable_upper, diagnostics=details)
    assert bound == 3
    assert details['reason'] == 'bounded' and details['pivots'] == 2


@pytest.mark.parametrize('options,reason', [({'max_pivots': 0}, 'pivot_limit'),
    ({'max_visits': 0}, 'work_limit'), ({'max_bits': 1}, 'bit_limit'),
    ({'time_limit_s': 1e-12}, 'time_limit')])
def test_recovery_explains_resource_stop(options, reason):
    p = coupled(); details = {}
    assert equality_residual_lower_bound(p, [Fraction(1), Fraction(0)],
        p.variable_lower, p.variable_upper, diagnostics=details, **options) is None
    assert details['reason'] == reason


@pytest.mark.parametrize('seed',range(10))
def test_exact_equality_basis_matches_constructed_unique_solution(seed):
    rng=np.random.default_rng(seed)
    a=np.eye(5,dtype=int)+np.tril(rng.integers(-3,4,size=(5,5)),-1)
    # Dense coupling without changing exact invertibility.
    a[0]+=2*a[-1]
    x=rng.integers(-4,5,5);c=rng.integers(-5,6,5)
    p=LinearProblem.from_data(A=a,c=c,variable_lower=[-np.inf]*5,variable_upper=[np.inf]*5,
        constraint_lower=a@x,constraint_upper=a@x)
    bound=equality_residual_lower_bound(p,[Fraction(int(v)) for v in c],p.variable_lower,p.variable_upper)
    assert bound == int(c@x)
