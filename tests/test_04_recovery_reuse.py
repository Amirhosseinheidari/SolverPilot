from dataclasses import replace
import numpy as np
import pytest
from solverpilot import LinearProblem, QuadraticProblem, solve
from solverpilot.backends import ScipyHighsLPBackend, OSQPNativeBackend
from solverpilot.validate import recover_lp_certificate, verify_lp_certificate


def test_infeasible_recovery_and_tampering():
    p = LinearProblem.from_data(
        A=[[1], [-1]],
        c=[1],
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[-np.inf, -np.inf],
        constraint_upper=[0, -1],
    )
    c = recover_lp_certificate(p, termination="infeasible")
    assert c.verified and verify_lp_certificate(p, c)
    assert not verify_lp_certificate(p, replace(c, dual=np.zeros(3)))
    assert not verify_lp_certificate(p, replace(c, problem_data_hash="other-model"))
    r = solve(p, backend=ScipyHighsLPBackend(), certificate_recovery=2.0)
    assert r.raw_statistics["termination_certificate"]["verified"]


def test_unbounded_recovery_and_no_false_infeasibility():
    p = LinearProblem.from_data(
        A=np.empty((0, 1)),
        c=[-1],
        variable_lower=[0],
        variable_upper=[np.inf],
        constraint_lower=[],
        constraint_upper=[],
    )
    c = recover_lp_certificate(p, termination="unbounded")
    assert c.verified and verify_lp_certificate(p, c)
    assert not verify_lp_certificate(p, replace(c, direction=-c.direction))
    assert not recover_lp_certificate(p, termination="infeasible").verified


def test_osqp_avoids_matrix_update_on_vector_only_change():
    pytest.importorskip("osqp")

    def qp(q, P=2):
        return QuadraticProblem.from_data(
            P=[[P]],
            q=[q],
            A=np.empty((0, 1)),
            variable_lower=[0],
            variable_upper=[3],
            constraint_lower=[],
            constraint_upper=[],
        )

    b = OSQPNativeBackend()
    solve(qp(-2), backend=b)
    second = solve(qp(-3), backend=b)
    assert second.raw_statistics["reuse_report"]["workspace"] == "observed"
    assert second.raw_statistics["reuse_report"]["matrix_update"] is False
    third = solve(qp(-3, 4), backend=b)
    assert third.raw_statistics["reuse_report"]["matrix_update"] is True
    assert third.raw_statistics["reuse_report"]["numeric_factorization"] == "unknown"
