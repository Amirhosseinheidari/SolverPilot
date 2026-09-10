from __future__ import annotations

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, Session
from solverpilot.backends import BackendRegistry, ScipyVendoredHighsDevBackend
from solverpilot.problem import VariableDomain
from solverpilot.validate import PublicStatus


pytestmark = pytest.mark.skipif(
    not ScipyVendoredHighsDevBackend().is_available(),
    reason="SciPy vendored HiGHS private bindings unavailable",
)


def _lp(cost=(1.0, 2.0), rhs=1.0):
    return LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=cost,
        variable_lower=[0.0, 0.0],
        variable_upper=[10.0, 10.0],
        constraint_lower=[rhs],
        constraint_upper=[np.inf],
    )


def test_vendored_highs_lp_basis_reuse_is_explicitly_reported():
    backend = ScipyVendoredHighsDevBackend(solver="simplex")
    first = backend.solve(_lp())
    second = backend.solve(_lp(cost=(2.0, 1.0), rhs=1.2))
    assert first.backend_status == "optimal"
    assert first.raw_statistics["reuse_applied"] is False
    assert second.backend_status == "optimal"
    assert second.raw_statistics["reuse_applied"] is True
    assert second.raw_statistics["reuse_mode"] == "explicit_basis_hot_start"
    assert second.raw_statistics["same_structure"] is True
    assert second.objective_reported == pytest.approx(1.2, abs=1e-8)


def test_vendored_highs_structural_change_forces_cold_rebuild():
    backend = ScipyVendoredHighsDevBackend(solver="simplex")
    backend.solve(_lp())
    changed = LinearProblem.from_data(
        A=[[1.0, 1.0], [1.0, 0.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[10.0, 10.0],
        constraint_lower=[1.0, 0.0],
        constraint_upper=[np.inf, 5.0],
    )
    result = backend.solve(changed)
    assert result.backend_status == "optimal"
    assert result.raw_statistics["same_structure"] is False
    assert result.raw_statistics["reuse_applied"] is False
    assert result.raw_statistics["reuse_mode"] == "cold_rebuild"


def test_vendored_highs_milp_injects_previous_incumbent():
    backend = ScipyVendoredHighsDevBackend()
    p1 = LinearProblem.from_data(
        A=[[1.0, 2.0]],
        c=[-1.0, -1.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf],
        constraint_upper=[2.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )
    p2 = LinearProblem.from_data(
        A=[[1.0, 2.0]],
        c=[-1.2, -1.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf],
        constraint_upper=[2.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )
    backend.solve(p1)
    result = backend.solve(p2)
    assert result.backend_status == "optimal"
    assert result.raw_statistics["reuse_applied"] is True
    assert result.raw_statistics["reuse_mode"] == "mip_solution_start"
    assert result.objective_reported == pytest.approx(-1.2, abs=1e-8)


def test_vendored_highs_qp_is_native_and_valid():
    q = QuadraticProblem.from_data(
        P=[[1.0]],
        A=np.zeros((0, 1)),
        q=[-2.0],
        variable_lower=[0.0],
        variable_upper=[10.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    backend = ScipyVendoredHighsDevBackend()
    result = backend.solve(q)
    assert result.backend_status == "optimal"
    assert result.x[0] == pytest.approx(2.0, abs=1e-5)
    assert result.objective_reported == pytest.approx(-2.0, abs=1e-8)


def test_session_only_marks_reuse_applied_after_backend_reports_it():
    registry = BackendRegistry()
    registry.register(ScipyVendoredHighsDevBackend(solver="simplex"))
    session = Session(_lp(), registry=registry)
    r0 = session.solve(backend="scipy-vendored-highs-native-dev")
    assert r0.status is PublicStatus.VALID_OPTIMAL
    session.update(c=np.array([2.0, 1.0]), constraint_lower=np.array([1.2]))
    r1 = session.solve(backend="scipy-vendored-highs-native-dev")
    assert r1.status is PublicStatus.VALID_OPTIMAL
    assert session.last_reuse_assessment is not None
    assert session.last_reuse_assessment.applied is True
    assert "explicit_basis_hot_start" in session.last_reuse_assessment.reason


def test_vendored_backend_is_not_registered_by_default():
    from solverpilot.runtime import default_registry

    registry = default_registry()
    assert "scipy-vendored-highs-native-dev" not in registry.names()


def test_vendored_highs_iis_verifier_finds_conflicting_rows():
    p = LinearProblem.from_data(
        A=[[1.0], [1.0]],
        c=[0.0],
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[1.0, -np.inf],
        constraint_upper=[np.inf, 0.0],
    )
    backend = ScipyVendoredHighsDevBackend()
    solve_result = backend.solve(p)
    assert solve_result.backend_status == "infeasible"
    iis = backend.compute_iis(p)
    assert iis.valid is True
    assert set(iis.row_indices) == {0, 1}


def test_vendored_highs_per_run_runtime_is_delta_not_cumulative():
    backend = ScipyVendoredHighsDevBackend(solver="simplex")
    first = backend.solve(_lp())
    second = backend.solve(_lp(cost=(1.1, 1.9)))
    assert first.raw_statistics["highs_run_delta_s"] >= 0.0
    assert second.raw_statistics["highs_run_delta_s"] >= 0.0
    assert second.raw_statistics["highs_runtime_cumulative_s"] >= first.raw_statistics["highs_runtime_cumulative_s"]
    # The delta must not simply equal the cumulative timer on a reused solver.
    assert second.raw_statistics["highs_run_delta_s"] <= second.raw_statistics["highs_runtime_cumulative_s"]
