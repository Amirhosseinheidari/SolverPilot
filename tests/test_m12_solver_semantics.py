from __future__ import annotations

import sys
from types import SimpleNamespace

import numpy as np

from solverpilot import LinearProblem, ObjectiveSense, QuadraticProblem
from solverpilot.backends.osqp_native import (
    _dual_certificate_check,
    _osqp_data,
    _osqp_status,
    _primal_certificate_check,
)
from solverpilot.backends.pyscipopt_native import PySCIPOptNativeBackend


def test_osqp_inaccurate_infeasibility_is_not_definitive():
    assert _osqp_status("primal infeasible inaccurate", False) == "infeasible_candidate"
    assert _osqp_status("dual infeasible inaccurate", False) == "unbounded_candidate"


def test_osqp_primal_certificate_check_accepts_valid_separator():
    p = QuadraticProblem.from_data(
        P=[[0.0]],
        A=[[1.0], [1.0]],
        q=[0.0],
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[1.0, -np.inf],
        constraint_upper=[np.inf, 0.0],
    )
    data = _osqp_data(p)
    # Two model rows + one identity-bound row appended by the OSQP adapter.
    check = _primal_certificate_check(data, np.array([-1.0, 1.0, 0.0]), eps=1e-8)
    assert check["valid"] is True
    assert check["stationarity_inf"] <= 1e-8
    assert check["separating_value"] < 0


def test_osqp_primal_certificate_check_rejects_wrong_separator():
    p = QuadraticProblem.from_data(
        P=[[0.0]], A=[[1.0]], q=[0.0],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    check = _primal_certificate_check(_osqp_data(p), np.array([1.0, 0.0]), eps=1e-8)
    assert check["valid"] is False


def test_osqp_dual_certificate_check_accepts_unbounded_direction():
    p = QuadraticProblem.from_data(
        P=[[0.0]], A=np.zeros((0, 1)), q=[-1.0],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[], constraint_upper=[],
    )
    check = _dual_certificate_check(p, _osqp_data(p), np.array([1.0]), eps=1e-8)
    assert check["valid"] is True
    assert check["q_dot_s"] < 0


def test_osqp_dual_certificate_check_enforces_row_signs():
    p = QuadraticProblem.from_data(
        P=[[0.0]], A=[[1.0]], q=[-1.0],
        variable_lower=[-np.inf], variable_upper=[np.inf],
        constraint_lower=[-np.inf], constraint_upper=[0.0],
    )
    # For an upper-only row, A s must be <= 0. s=1 violates that requirement.
    check = _dual_certificate_check(p, _osqp_data(p), np.array([1.0]), eps=1e-8)
    assert check["valid"] is False
    assert check["row_sign_violation"] > 0


class _Expr:
    def __add__(self, other): return self
    __radd__ = __add__
    def __mul__(self, other): return self
    __rmul__ = __mul__
    def __ge__(self, other): return self
    def __le__(self, other): return self
    def __eq__(self, other): return self


class _Var(_Expr):
    pass


class _FakeSCIPModel:
    last = None
    def __init__(self, name):
        type(self).last = self
        self.offset = 0.0
        self.objective_sense = None
    def hideOutput(self): pass
    def setParam(self, *args): pass
    def addVar(self, **kwargs): return _Var()
    def setObjective(self, expr, sense): self.objective_sense = sense
    def addObjoffset(self, value): self.offset += float(value)
    def addCons(self, *args, **kwargs): pass


def test_pyscipopt_builder_preserves_objective_offset(monkeypatch):
    fake = SimpleNamespace(Model=_FakeSCIPModel, quicksum=lambda it: _Expr())
    monkeypatch.setitem(sys.modules, "pyscipopt", fake)
    monkeypatch.setattr(PySCIPOptNativeBackend, "is_available", lambda self: True)
    p = LinearProblem.from_data(
        A=[[1.0]], c=[2.0], variable_lower=[0.0], variable_upper=[1.0],
        constraint_lower=[0.0], constraint_upper=[1.0],
        objective_offset=7.5, objective_sense=ObjectiveSense.MINIMIZE,
    )
    PySCIPOptNativeBackend()._build_model(p)
    assert _FakeSCIPModel.last is not None
    assert _FakeSCIPModel.last.offset == 7.5
    assert _FakeSCIPModel.last.objective_sense == "minimize"
