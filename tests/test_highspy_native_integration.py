import numpy as np
import pytest

highspy = pytest.importorskip("highspy", reason="optional native HiGHS integration")

from solverpilot import LinearProblem, QuadraticProblem, VariableDomain, execute
from solverpilot.backends import HighspyNativeBackend
from solverpilot.validate import PublicStatus


def test_highspy_native_lp():
    p = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    r = execute(p, HighspyNativeBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.objective == pytest.approx(1.0, abs=1e-7)


def test_highspy_native_milp():
    p = LinearProblem.from_data(
        A=[[2.0, 1.0]], c=[-3.0, -2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[2.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )
    r = execute(p, HighspyNativeBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.objective == pytest.approx(-3.0, abs=1e-7)


def test_highspy_native_qp():
    p = QuadraticProblem.from_data(
        P=[[2.0, 0.0], [0.0, 4.0]],
        A=np.zeros((0, 2)), q=[-2.0, -4.0],
        variable_lower=[-5.0, -5.0], variable_upper=[5.0, 5.0],
        constraint_lower=[], constraint_upper=[],
    )
    r = execute(p, HighspyNativeBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert np.allclose(r.x, [1.0, 1.0], atol=1e-6)
