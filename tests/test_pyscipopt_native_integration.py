import numpy as np
import pytest

pyscipopt = pytest.importorskip("pyscipopt", reason="optional native SCIP integration")

from solverpilot import LinearProblem, VariableDomain, execute
from solverpilot.backends import PySCIPOptNativeBackend
from solverpilot.validate import PublicStatus


def test_pyscipopt_native_milp():
    p = LinearProblem.from_data(
        A=[[2.0, 1.0]], c=[-3.0, -2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf], constraint_upper=[2.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
    )
    r = execute(p, PySCIPOptNativeBackend())
    assert r.status is PublicStatus.VALID_OPTIMAL
    assert r.objective == pytest.approx(-3.0, abs=1e-7)


def test_pyscipopt_iis():
    p = LinearProblem.from_data(
        A=[[1.0], [1.0]], c=[0.0],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[1.0, -np.inf], constraint_upper=[np.inf, 0.0],
    )
    iis = PySCIPOptNativeBackend().generate_iis(p)
    assert iis.constraint_names
    assert iis.irreducible
