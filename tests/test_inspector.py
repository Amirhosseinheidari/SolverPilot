import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem, QuadraticProblem, VariableDomain
from solverpilot.inspect import inspect_problem


def test_lp_fingerprint_core_statistics():
    problem = LinearProblem.from_data(
        A=sparse.csr_matrix([[1.0, 0.0, -100.0], [0.0, 2.0, 0.0]]),
        c=[1.0, 0.0, 10.0],
        variable_lower=[0.0, -np.inf, 0.0],
        variable_upper=[1.0, np.inf, 5.0],
        constraint_lower=[0.0, 2.0],
        constraint_upper=[10.0, 2.0],
        domains=[VariableDomain.BINARY, VariableDomain.CONTINUOUS, VariableDomain.INTEGER],
    )
    fp = inspect_problem(problem)
    assert fp.problem_class == "milp"
    assert fp.n_variables == 3
    assert fp.n_constraints == 2
    assert fp.nnz_a == 3
    assert fp.density_a == pytest.approx(0.5)
    assert fp.binary_fraction == pytest.approx(1 / 3)
    assert fp.integer_fraction == pytest.approx(2 / 3)
    assert fp.equality_fraction == pytest.approx(0.5)
    assert fp.fully_bounded_variable_fraction == pytest.approx(2 / 3)
    assert fp.free_variable_fraction == pytest.approx(1 / 3)
    assert fp.coefficient_dynamic_range_log10 == pytest.approx(2.0)
    assert fp.objective_dynamic_range_log10 == pytest.approx(1.0)
    assert fp.singleton_rows == 1
    assert fp.singleton_cols == 3
    assert fp.positive_coefficient_fraction == pytest.approx(2 / 3)
    assert fp.negative_coefficient_fraction == pytest.approx(1 / 3)
    assert fp.estimated_numeric_bytes > 0


def test_qp_fingerprint():
    problem = QuadraticProblem.from_data(
        P=np.diag([2.0, 4.0]),
        A=[[1.0, 1.0]],
        q=[-1.0, -2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[np.inf, np.inf],
        constraint_lower=[-np.inf],
        constraint_upper=[3.0],
    )
    fp = inspect_problem(problem)
    assert fp.problem_class == "convex_qp"
    assert fp.quadratic_nnz == 2
    assert fp.quadratic_density == pytest.approx(0.5)
    assert fp.convexity_status == "confirmed"


def test_zero_coefficient_dynamic_range_is_none():
    problem = LinearProblem.from_data(
        A=np.zeros((0, 2)), c=[0.0, 0.0],
        variable_lower=[-np.inf, -np.inf], variable_upper=[np.inf, np.inf],
        constraint_lower=[], constraint_upper=[],
    )
    fp = inspect_problem(problem)
    assert fp.coefficient_dynamic_range_log10 is None
    assert fp.objective_dynamic_range_log10 is None
    assert fp.density_a == 0.0
    assert fp.row_nnz.mean == 0.0
