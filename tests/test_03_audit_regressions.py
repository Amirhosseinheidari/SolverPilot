"""Permanent regressions for the independently reproduced release 0.2 defects."""

import numpy as np
import pytest
import solverpilot as sp
from solverpilot.validate import verify_infeasibility, verify_optimality, verify_unboundedness


def test_farkas_check_rejects_a_feasible_problem():
    p = sp.LinearProblem.from_data(
        A=[[1e-10]],
        c=[0.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )
    assert sp.validate_solution(p, sp.CandidateSolution(np.array([1e10]))).valid
    assert not verify_infeasibility(p, [-1.0, 0.0]), "x=1e10 is a feasible witness"


def test_recession_check_rejects_a_bounded_problem():
    p = sp.LinearProblem.from_data(
        A=[[1e-10]],
        c=[-1.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[-np.inf],
        constraint_upper=[1.0],
    )
    assert not verify_unboundedness(p, [0.0], [1.0]), "The objective is bounded below by -1e10"


def test_optimality_gap_accounts_for_dual_stationarity_error():
    p = sp.LinearProblem.from_data(
        A=np.empty((0, 2)),
        c=[1e-10, -1e-10],
        variable_lower=[0.0, 0.0],
        variable_upper=[1e10, 1e10],
        constraint_lower=[],
        constraint_upper=[],
    )
    assert not verify_optimality(p, [0.0, 0.0], [0.0, 0.0]).verified, (
        "Objective 0 is worse than the feasible objective -1"
    )


def test_unsetting_osqp_max_iter_restores_default():
    pytest.importorskip("osqp")
    from solverpilot.backends import OSQPNativeBackend

    p = sp.QuadraticProblem.from_data(
        P=[[2.0]],
        A=np.empty((0, 1)),
        q=[-4.0],
        variable_lower=[0.0],
        variable_upper=[10.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    b = OSQPNativeBackend(max_iter=1)
    sp.execute(p, b)
    b.max_iter = None
    sp.execute(p, b)
    fresh = OSQPNativeBackend()
    sp.execute(p, fresh)
    assert b._solver.settings.max_iter == fresh._solver.settings.max_iter


def test_named_values_reject_other_conic_problem_results():
    pytest.importorskip("clarabel")
    from solverpilot.conic import ClarabelBackend
    from solverpilot.model import named_values

    rows = []
    for value in (1.0, 2.0):
        m = sp.Model()
        x = m.variable(1, lower=value, upper=value, name="fixed_x")
        t = m.variable(lower=0.0, upper=10.0, name="radius")
        m.soc(t, x)
        m.minimize(t)
        compiled = m.compile()
        rows.append((m, compiled, compiled.solve(backend=ClarabelBackend())))
    with pytest.raises(ValueError):
        named_values(rows[0][0], rows[0][1], rows[1][2])


def test_nlp_result_cannot_change_after_validation():
    pytest.importorskip("casadi")
    m = sp.Model()
    x = m.variable(lower=-1.0, upper=1.0)
    m.minimize(x.exp())
    r = m.compile().solve()
    assert r.validation.valid
    with pytest.raises(ValueError):
        r.x[0] = 999.0
