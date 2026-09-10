from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain, execute
from solverpilot.backends import (
    CasadiCBCBridgeBackend,
    CasadiHighsBridgeBackend,
    CasadiOSQPBridgeBackend,
    probe_backend,
)


@pytest.mark.parametrize("backend", [CasadiOSQPBridgeBackend(), CasadiHighsBridgeBackend(), CasadiCBCBridgeBackend()])
def test_casadi_bridge_health(backend):
    if not backend.is_available():
        pytest.skip("CasADi/plugin unavailable")
    report = probe_backend(backend)
    assert report.healthy, report


def test_casadi_osqp_qp_exact():
    b = CasadiOSQPBridgeBackend()
    if not b.is_available(): pytest.skip("CasADi OSQP unavailable")
    p = QuadraticProblem.from_data(
        P=[[4.0, 0.0], [0.0, 2.0]], A=[[1.0, 1.0]], q=[-8.0, -2.0],
        variable_lower=[0.0, 0.0], variable_upper=[10.0, 10.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    r = execute(p, b)
    assert r.validation and r.validation.valid
    assert r.status.value == "valid_optimal"


def test_casadi_cbc_binary_milp():
    b=CasadiCBCBridgeBackend()
    if not b.is_available(): pytest.skip("CasADi CBC unavailable")
    p=LinearProblem.from_data(
        A=[[1,1,1]],c=[3,2,1],variable_lower=[0,0,0],variable_upper=[1,1,1],
        constraint_lower=[-np.inf],constraint_upper=[1],
        domains=[VariableDomain.BINARY]*3, objective_sense=ObjectiveSense.MAXIMIZE,
    )
    r=execute(p,b)
    assert r.status.value == "valid_optimal"
    assert np.isclose(r.objective,3.0)


def test_casadi_osqp_rejects_integer():
    b=CasadiOSQPBridgeBackend()
    if not b.is_available(): pytest.skip("CasADi OSQP unavailable")
    p=LinearProblem.from_data(A=[[1]],c=[1],variable_lower=[0],variable_upper=[1],constraint_lower=[0],constraint_upper=[1],domains=[VariableDomain.BINARY])
    with pytest.raises(ValueError): b.solve(p)


def test_casadi_bridges_infeasible_status():
    p=LinearProblem.from_data(
        A=[[1],[1]],c=[1],variable_lower=[-10],variable_upper=[10],
        constraint_lower=[1,-np.inf],constraint_upper=[np.inf,0],
    )
    for b in (CasadiHighsBridgeBackend(), CasadiCBCBridgeBackend()):
        if not b.is_available(): continue
        r=execute(p,b)
        assert r.status.value in {"infeasible", "infeasible_or_unbounded"}
        assert r.x is None

def test_casadi_osqp_reports_real_primal_warm_start_on_same_structure():
    b=CasadiOSQPBridgeBackend()
    if not b.is_available(): pytest.skip("CasADi OSQP unavailable")
    p1=QuadraticProblem.from_data(P=[[2.0]],A=sparse.csr_matrix((0,1)),q=[-2.0],variable_lower=[-10],variable_upper=[10],constraint_lower=[],constraint_upper=[])
    p2=QuadraticProblem.from_data(P=[[2.0]],A=sparse.csr_matrix((0,1)),q=[-4.0],variable_lower=[-10],variable_upper=[10],constraint_lower=[],constraint_upper=[])
    r1=execute(p1,b); r2=execute(p2,b)
    assert r1.trace.reuse_applied is False
    assert r2.trace.reuse_applied is True
    assert r2.trace.reuse_mode == "casadi_osqp_primal_warm_start"
    assert r2.validation and r2.validation.valid

def test_casadi_budget_mapping_is_not_silently_ignored():
    from solverpilot.plan import SolveBudget
    from solverpilot.runtime.budgeting import apply_budget
    with pytest.raises(NotImplementedError, match="no verified wall-time budget mapping"):
        apply_budget(CasadiOSQPBridgeBackend(), SolveBudget(wall_time_s=0.1))
    with pytest.raises(NotImplementedError, match="no verified wall-time budget mapping"):
        apply_budget(CasadiCBCBridgeBackend(), SolveBudget(wall_time_s=0.1))
    b=apply_budget(CasadiHighsBridgeBackend(), SolveBudget(wall_time_s=0.1))
    assert b.time_limit_s == 0.1
