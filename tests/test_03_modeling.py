import numpy as np
import pytest
from scipy import sparse
from solverpilot.model import Model, abs, norm, huber, log_sum_exp, quad_form, named_values


@pytest.mark.parametrize("shape", [(5,), (3, 5)])
def test_sparse_affine_matches_numeric_matmul(shape):
    rng = np.random.default_rng(730)
    m = Model()
    x = m.variable(shape, lower=-2, upper=2)
    value = rng.normal(size=shape)
    if len(shape) == 1:
        A = sparse.random(4, 5, density=0.4, random_state=rng, format="csr")
        expression = m.constant(A) @ x
        expected = A @ value
    else:
        A = sparse.random(5, 4, density=0.4, random_state=rng, format="csr")
        expression = x @ m.constant(A)
        expected = value @ A
    m.add(expression <= 100)
    m.minimize(expression.sum())
    compiled = m.compile()
    assert compiled.execution_ir.A @ value.reshape(-1) == pytest.approx(expected.reshape(-1))
    assert compiled.execution_ir.c @ value.reshape(-1) == pytest.approx(expected.sum())
    assert m.clone().compile().execution_ir.data_hash == compiled.execution_ir.data_hash


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda x: abs(x).sum(), 7.0),
        (lambda x: norm(x, 1), 7.0),
        (lambda x: norm(x), 5.0),
        (lambda x: norm(x, np.inf), 4.0),
        (lambda x: huber(x).sum(), 12.0),
        (lambda x: log_sum_exp(x), float(np.logaddexp(3.0, -4.0))),
        (lambda x: quad_form(x, np.eye(2)), 25.0),
    ],
)
def test_atoms_analytic_values_and_reconstruction(factory, expected):
    pytest.importorskip("clarabel")
    m = Model()
    x = m.variable(2, lower=[3.0, -4.0], upper=[3.0, -4.0], name="original")
    m.minimize(factory(x))
    compiled = m.compile()
    result = compiled.solve()
    assert result.validation.valid
    objective = getattr(result, "objective", getattr(result, "objective_reported", None))
    assert objective == pytest.approx(expected, abs=2e-6)
    assert compiled.reconstruct_primal(result.x) == pytest.approx([3.0, -4.0])
    assert len(named_values(m, compiled, result)) == 2
    assert m.clone().compile().execution_ir.data_hash == compiled.execution_ir.data_hash


def test_atoms_reject_wrong_direction_and_nonconvex_composition():
    m = Model()
    x = m.variable()
    m.maximize(abs(x))
    with pytest.raises(ValueError, match="direction"):
        m.compile()
    m.minimize(-norm(x))
    with pytest.raises(ValueError, match="direction"):
        m.compile()
    m.minimize(x)
    m.add(abs(x) >= 2)
    with pytest.raises(ValueError, match="direction"):
        m.compile()


def test_atom_constraint_and_parameter_update():
    pytest.importorskip("clarabel")
    m = Model()
    x = m.variable(2, lower=-10, upper=10)
    radius = m.parameter(value=1.0, name="radius")
    m.add(norm(x) <= radius)
    m.minimize(-x[0] - x[1])
    for value in (1.0, 2.0, 1.0):
        radius.value = value
        compiled = m.compile()
        result = compiled.solve()
        assert result.validation.valid
        assert result.objective_reported == pytest.approx(-np.sqrt(2) * value, abs=1e-6)
        assert compiled.validate_original(m, result.x, atol=1e-6).valid


def test_sparse_matmul_does_not_densify_constant(monkeypatch):
    m = Model()
    x = m.variable(1000, lower=0, upper=1)
    A = sparse.eye(1000, format="csr")
    m.add(m.constant(A) @ x <= 1)
    m.minimize(x.sum())

    def forbidden(*args, **kwargs):
        raise AssertionError("sparse constant was densified")

    monkeypatch.setattr(sparse.csr_matrix, "toarray", forbidden)
    assert m.compile().execution_ir.A.nnz == 1000
