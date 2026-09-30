from fractions import Fraction
import numpy as np
import pytest
from scipy import sparse
from solverpilot.validate._certificate_arithmetic import dot, matvec
from solverpilot.validate.optimality import verify_optimality
from solverpilot import LinearProblem


@pytest.mark.parametrize('seed', range(12))
def test_dyadic_accumulator_matches_fraction_reference_bit_for_bit(seed):
    rng = np.random.default_rng(seed)
    a = np.ldexp(rng.uniform(-1, 1, (12, 20)), rng.integers(-1000, 1000, (12, 20)))
    b = np.ldexp(rng.uniform(-1, 1, 20), rng.integers(-1000, 1000, 20))
    expected = [sum((Fraction(float(x))*Fraction(float(y)) for x,y in zip(row,b)), Fraction()) for row in a]
    assert matvec(sparse.csr_matrix(a), b) == expected
    for row, reference in zip(a, expected):
        assert dot(row, b) == reference


def test_cancellation_subnormal_and_overflow_products_are_exact():
    smallest = np.nextafter(0., 1.)
    assert dot([1e308, 1., -1e308], [1e308, smallest, 1e308]) == Fraction(float(smallest))
    assert dot([smallest], [smallest]) == Fraction(float(smallest))**2
    assert dot([], []) == 0


def test_invalid_primal_or_stationarity_never_enters_expensive_arithmetic(monkeypatch):
    import solverpilot.validate.optimality as module
    def forbidden(*a, **k): raise AssertionError('expensive stage reached')
    monkeypatch.setattr(module, 'matvec', forbidden)
    p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],variable_upper=[2.],
                               constraint_lower=[1.],constraint_upper=[2.])
    assert not verify_optimality(p, [0.], [-1., 0.], fast_reject=True).verified
    assert not verify_optimality(p, [1.], [0., 0.], fast_reject=True).verified
