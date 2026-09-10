import numpy as np
import pytest

from solverpilot import ConvexityStatus, QuadraticProblem


def qp(P, verify=True):
    return QuadraticProblem.from_data(
        P=P,
        A=[[1.0, 1.0]],
        q=[-1.0, -1.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[np.inf, np.inf],
        constraint_lower=[1.0],
        constraint_upper=[1.0],
        verify_convexity=verify,
    )


def test_psd_qp_confirmed():
    p = qp([[2.0, 0.0], [0.0, 1.0]])
    assert p.convexity_status is ConvexityStatus.CONFIRMED


def test_nonconvex_qp_rejected():
    with pytest.raises(ValueError, match="positive semidefinite"):
        qp([[1.0, 0.0], [0.0, -1.0]])


def test_unknown_when_verification_disabled():
    p = qp([[2.0, 0.0], [0.0, 1.0]], verify=False)
    assert p.convexity_status is ConvexityStatus.UNKNOWN


def test_maximize_rejected_for_v01():
    with pytest.raises(ValueError, match="convex minimization"):
        QuadraticProblem.from_data(
            P=np.eye(1),
            A=[[1.0]],
            q=[1.0],
            variable_lower=[0.0],
            variable_upper=[1.0],
            constraint_lower=[0.0],
            constraint_upper=[1.0],
            objective_sense="maximize",
        )
