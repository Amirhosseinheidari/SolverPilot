import numpy as np
import pytest

osqp = pytest.importorskip("osqp", reason="optional native OSQP integration")

from solverpilot import QuadraticProblem, Session
from solverpilot.backends import BackendRegistry, OSQPNativeBackend
from solverpilot.validate import PublicStatus


def make_qp(q):
    return QuadraticProblem.from_data(
        P=[[4.0, 0.0], [0.0, 2.0]],
        A=[[1.0, 1.0]],
        q=q,
        variable_lower=[-2.0, -2.0],
        variable_upper=[2.0, 2.0],
        constraint_lower=[-1.0],
        constraint_upper=[1.0],
    )


def test_osqp_native_reuses_same_sparsity_session():
    backend = OSQPNativeBackend(eps_abs=1e-8, eps_rel=1e-8)
    reg = BackendRegistry()
    reg.register(backend)
    session = Session(make_qp([-1.0, 0.5]), registry=reg)
    r0 = session.solve(backend=backend)
    assert r0.status in {PublicStatus.VALID_OPTIMAL, PublicStatus.VALID_FEASIBLE}
    session.update(q=[-0.8, 0.25])
    r1 = session.solve(backend=backend)
    assert r1.status in {PublicStatus.VALID_OPTIMAL, PublicStatus.VALID_FEASIBLE}
    assert (r1.raw_statistics or {}).get("reuse_applied") is True
    assert session.last_reuse_assessment is not None
    assert session.last_reuse_assessment.applied
