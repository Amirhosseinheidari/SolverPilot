import numpy as np
import pytest
from scipy import sparse
from solverpilot import LinearProblem, QuadraticProblem, solve, SolveBudget
from solverpilot.backends import PDLPBackend

pytestmark = pytest.mark.native


@pytest.mark.parametrize("quadratic", [False, True])
def test_pdlp_thousand_variables_and_independent_dual(quadratic):
    pytest.importorskip("ortools")
    n = 1000
    kwargs = dict(
        A=sparse.eye(n),
        variable_lower=np.zeros(n),
        variable_upper=np.ones(n) * 2,
        constraint_lower=np.ones(n) * 0.5,
        constraint_upper=np.ones(n) * 2,
    )
    p = (
        QuadraticProblem.from_data(P=sparse.eye(n) * 2, q=np.ones(n) * -2, **kwargs)
        if quadratic
        else LinearProblem.from_data(c=np.ones(n), **kwargs)
    )
    r = solve(p, backend=PDLPBackend(), budget=SolveBudget(wall_time_s=20, threads=1))
    assert r.validation.valid
    assert r.objective == pytest.approx(-1000 if quadratic else 500, abs=1e-6)
    assert r.optimality_evidence.independently_verified_optimal
    assert r.raw_statistics["isolated_worker"]


def test_maximize_dual_sign_and_offset():
    pytest.importorskip("ortools")
    p = LinearProblem.from_data(
        A=[[1.0]],
        c=[2.0],
        variable_lower=[-2.0],
        variable_upper=[5.0],
        constraint_lower=[0.0],
        constraint_upper=[3.0],
        objective_sense="maximize",
        objective_offset=7,
    )
    r = solve(p, backend="ortools-pdlp")
    assert r.validation.valid and r.objective == pytest.approx(13, abs=1e-6)
    assert r.optimality_evidence.independently_verified_optimal


def test_nondiagonal_and_integer_fail_closed():
    pytest.importorskip("ortools")
    p = QuadraticProblem.from_data(
        P=[[2, 1], [1, 2]],
        A=np.empty((0, 2)),
        q=[0, 0],
        variable_lower=[-1, -1],
        variable_upper=[1, 1],
        constraint_lower=[],
        constraint_upper=[],
    )
    with pytest.raises(ValueError, match="diagonal"):
        PDLPBackend().solve(p)
    p = LinearProblem.from_data(
        A=[[1]],
        c=[1],
        variable_lower=[0],
        variable_upper=[1],
        constraint_lower=[0],
        constraint_upper=[1],
        domains=["binary"],
    )
    with pytest.raises(ValueError, match="integer"):
        PDLPBackend().solve(p)


def test_worker_timeout_is_not_success():
    pytest.importorskip("ortools")
    p = LinearProblem.from_data(
        A=[[1]],
        c=[1],
        variable_lower=[0],
        variable_upper=[1],
        constraint_lower=[0],
        constraint_upper=[1],
    )
    r = PDLPBackend(time_limit_s=1e-6).solve(p)
    assert r.backend_status == "limit_no_solution" and r.x is None


def test_dual_infeasible_report_is_not_a_primal_unbounded_proof(monkeypatch):
    import json
    from types import SimpleNamespace
    import solverpilot.backends.pdlp as adapter

    # Infeasible primal x<=0, x>=1; a separate free improving coordinate
    # also makes the dual infeasible. A dual ray cannot settle primal status.
    p = LinearProblem.from_data(
        A=[[1, 0], [-1, 0]],
        c=[0, -1],
        variable_lower=[-np.inf, 0],
        variable_upper=[np.inf, np.inf],
        constraint_lower=[-np.inf, -np.inf],
        constraint_upper=[0, -1],
    )
    payload = dict(
        protocol=adapter.PROTOCOL,
        termination="TERMINATION_REASON_DUAL_INFEASIBLE",
        iterations=10,
        x=[0, 1],
        dual=[0, 0],
        reduced_costs=[0, 0],
        corrected_dual_objective=None,
    )
    monkeypatch.setattr(PDLPBackend, "is_available", lambda _: True)
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(
            returncode=0, stderr="", stdout="SOLVERPILOT_PDLP_RESULT=" + json.dumps(payload)
        ),
    )
    r = solve(p, backend=PDLPBackend())
    assert r.backend_status == "infeasible_or_unbounded" and r.x is None
    assert not r.optimality_evidence.independently_verified_optimal


def test_native_dual_ray_can_be_resolved_by_independent_recovery():
    pytest.importorskip("ortools")
    p = LinearProblem.from_data(
        A=np.empty((0, 1)),
        c=[-1],
        variable_lower=[0],
        variable_upper=[np.inf],
        constraint_lower=[],
        constraint_upper=[],
    )
    r = solve(p, backend=PDLPBackend(), certificate_recovery=2.0)
    assert r.backend_status == "infeasible_or_unbounded"
    c = r.raw_statistics["termination_certificate"]
    assert c["verified"] and c["kind"] == "unbounded"
