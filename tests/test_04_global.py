import numpy as np
import pytest
from solverpilot.model import Model, Interval, abs as model_abs
from solverpilot.globalopt import (
    GlobalQuadraticProblem,
    SCIPGlobalBackend,
    solve_global,
    absolute,
    maximum,
    minimum,
)
from solverpilot.runtime import solve_any, solve_verified, UnverifiedSolutionError

pytestmark = pytest.mark.native


@pytest.fixture
def backend():
    pytest.importorskip("pyscipopt")
    return SCIPGlobalBackend(time_limit_s=20)


def test_matrix_nonconvex_and_miqp(backend):
    p = GlobalQuadraticProblem.from_data(
        P=[[0, -1], [-1, 0]],
        A=np.empty((0, 2)),
        q=[0, 0],
        variable_lower=[-2, -2],
        variable_upper=[2, 2],
        constraint_lower=[],
        constraint_upper=[],
    )
    summary, r = solve_any(p, backend=backend)
    assert summary.feasible and r.objective == pytest.approx(-4)
    assert summary.optimality == "solver_reported"
    with pytest.raises(UnverifiedSolutionError):
        solve_verified(p, backend=backend)
    p = GlobalQuadraticProblem.from_data(
        P=np.eye(2) * 2,
        A=np.empty((0, 2)),
        q=[-4.6, -1.4],
        variable_lower=[0, 0],
        variable_upper=[4, 3],
        constraint_lower=[],
        constraint_upper=[],
        domains=["integer", "continuous"],
        objective_offset=2.3**2 + 0.7**2,
    )
    r = solve_global(p, backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(0.09, abs=1e-7)
    assert r.x[0] == pytest.approx(2)


def test_maximization_offset_and_rows(backend):
    p = GlobalQuadraticProblem.from_data(
        P=[[-2]],
        A=[[1]],
        q=[2],
        variable_lower=[0],
        variable_upper=[4],
        constraint_lower=[0],
        constraint_upper=[3],
        objective_sense="maximize",
        objective_offset=7,
    )
    r = solve_global(p, backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(8, abs=1e-7)


def test_global_minlp_and_nonlinear_equalities(backend):
    m = Model()
    z = m.variable(lower=0, upper=4, domain="integer")
    x = m.variable(lower=-2, upper=2)
    m.minimize((x * x - 1) ** 2 + 0.1 * x + (z - 2.3) ** 2)
    r = m.solve(backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(-0.010617376638, abs=1e-7)
    m = Model()
    x = m.variable(lower=-2, upper=2)
    m.add(x * x == 1)
    m.minimize(x)
    assert m.compile(target="global").solve(backend=backend).objective == pytest.approx(-1)
    m = Model()
    x = m.variable(lower=-2, upper=2)
    m.add_in_set(x * x, Interval(0.25, 1))
    m.minimize(x)
    assert m.solve(target="global", backend=backend).objective == pytest.approx(-1, abs=1e-7)


@pytest.mark.parametrize("atom", [model_abs, absolute])
def test_nonsmooth_global_abs(backend, atom):
    m = Model()
    x = m.variable(lower=-2, upper=2)
    m.minimize(-atom(x))
    r = m.solve(backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(-2)


def test_general_nonsmooth_compositions(backend):
    m = Model()
    x = m.variable(lower=-2, upper=2)
    m.minimize(-absolute(x * x - 1))
    assert m.solve(backend=backend).objective == pytest.approx(-3)
    m = Model()
    x = m.variable(2, lower=[-2, -1], upper=[2, 1])
    m.maximize(maximum(x) + minimum(x, 0).sum())
    r = m.solve(backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(2, abs=1e-7)


@pytest.mark.parametrize("active_value", [0, 1])
def test_indicator_composition_uses_bound_and_validates_original(backend, active_value):
    m = Model()
    x = m.variable(lower=-2, upper=2)
    z = m.variable(lower=0, upper=1, domain="binary")
    m.indicator(z, x * x <= 1, active_value=active_value)
    m.maximize(x + 0.5 * (z if active_value else 1 - z))
    compiled = m.compile(target="global")
    r = compiled.solve(backend=backend)
    assert r.validation.valid and r.objective == pytest.approx(2, abs=1e-7)
    assert r.raw_statistics["transformations"][0]["M"] >= 3
    assert compiled.validate_original(m, r.x).valid
    bad = np.array([2.0, float(active_value)])
    assert not compiled.validate_original(m, bad).valid


def test_domain_guard_and_parameter_snapshot(backend):
    m = Model()
    x = m.variable(lower=-1, upper=1)
    m.minimize(x.log())
    with pytest.raises(ValueError, match="domain"):
        m.compile(target="global")
    m = Model()
    x = m.variable(lower=-2, upper=2)
    p = m.parameter(value=0.1)
    m.minimize((x * x - 1) ** 2 + p * x)
    old = m.compile(target="global")
    p.value = -0.1
    new = old.refresh(m)
    assert new.data_hash != old.data_hash
    assert old.solve(backend=backend).x[0] < 0 and new.solve(backend=backend).x[0] > 0


def test_backend_capability_is_not_an_exact_proof(backend):
    caps = backend.capabilities()
    assert caps["available"] and caps["matrix_miqp"]
    assert not caps["independent_global_proof"]
