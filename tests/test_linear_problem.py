import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem, ObjectiveSense, VariableDomain


def base_problem(**overrides):
    data = dict(
        A=sparse.csr_matrix([[1.0, 2.0], [-1.0, 1.0]]),
        c=[1.0, 3.0],
        variable_lower=[0.0, -np.inf],
        variable_upper=[1.0, np.inf],
        constraint_lower=[-np.inf, 0.0],
        constraint_upper=[3.0, np.inf],
        domains=[VariableDomain.CONTINUOUS, VariableDomain.CONTINUOUS],
    )
    data.update(overrides)
    return LinearProblem.from_data(**data)


def test_constructs_and_normalizes_sparse_matrix():
    p = base_problem()
    assert sparse.isspmatrix_csr(p.A)
    assert p.n_variables == 2
    assert p.n_constraints == 2
    assert p.nnz == 4


def test_arrays_are_readonly():
    p = base_problem()
    with pytest.raises(ValueError):
        p.c[0] = 9.0
    with pytest.raises(ValueError):
        p.A.data[0] = 9.0


def test_bad_variable_bound_rejected():
    with pytest.raises(ValueError, match="inconsistent variable bounds"):
        base_problem(variable_lower=[2.0, 0.0], variable_upper=[1.0, 1.0])


def test_bad_constraint_bound_rejected():
    with pytest.raises(ValueError, match="inconsistent constraint bounds"):
        base_problem(constraint_lower=[4.0, 0.0], constraint_upper=[3.0, 1.0])


def test_nan_rejected():
    with pytest.raises(ValueError, match="must not contain NaN"):
        base_problem(c=[np.nan, 1.0])


def test_inf_matrix_coefficient_rejected():
    with pytest.raises(ValueError, match="A coefficients must be finite"):
        base_problem(A=[[np.inf, 0.0], [0.0, 1.0]])


def test_binary_bounds_must_live_inside_unit_interval():
    with pytest.raises(ValueError, match="binary variable"):
        base_problem(
            domains=[VariableDomain.BINARY, VariableDomain.CONTINUOUS],
            variable_lower=[-1.0, 0.0],
            variable_upper=[1.0, 1.0],
        )


def test_structural_hash_ignores_numeric_changes_but_data_hash_does_not():
    p1 = base_problem()
    p2 = base_problem(A=[[10.0, 2.0], [-1.0, 5.0]], c=[7.0, 3.0])
    assert p1.structural_hash == p2.structural_hash
    assert p1.data_hash != p2.data_hash


def test_structural_hash_changes_with_sparsity():
    p1 = base_problem()
    p2 = base_problem(A=[[1.0, 0.0], [-1.0, 1.0]])
    assert p1.structural_hash != p2.structural_hash


def test_objective_sense_parses_string():
    p = base_problem(objective_sense="maximize")
    assert p.objective_sense is ObjectiveSense.MAXIMIZE


def test_hash_is_stable_across_sparse_formats_and_duplicate_entries():
    A_coo = sparse.coo_matrix(
        ([0.5, 0.5, 2.0, -1.0, 1.0], ([0, 0, 0, 1, 1], [0, 0, 1, 0, 1])),
        shape=(2, 2),
    )
    p1 = base_problem(A=A_coo)
    p2 = base_problem(A=sparse.csc_matrix([[1.0, 2.0], [-1.0, 1.0]]))
    assert p1.structural_hash == p2.structural_hash
    assert p1.data_hash == p2.data_hash


def test_shape_mismatch_rejected():
    with pytest.raises(ValueError, match="c must have shape"):
        base_problem(c=[1.0])
