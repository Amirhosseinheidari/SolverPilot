"""GPU transport contracts run on CPU-only CI; these are not GPU qualification."""

import json
from types import SimpleNamespace

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, SolveBudget, solve
from solverpilot.backends import CuOptBackend
from solverpilot.backends.base import BackendUnavailableError
import solverpilot.backends.cuopt as adapter


def lp(**overrides):
    args = dict(
        A=[[1.0]],
        c=[2.0],
        variable_lower=[0.0],
        variable_upper=[5.0],
        constraint_lower=[1.0],
        constraint_upper=[3.0],
    )
    args.update(overrides)
    return LinearProblem.from_data(**args)


@pytest.fixture
def worker(monkeypatch):
    monkeypatch.setattr(CuOptBackend, "is_available", lambda _: True)
    payload = dict(
        protocol=adapter.PROTOCOL,
        gpu_count=1,
        solved_by="PDLP",
        termination="Optimal",
        x=[1.0],
        dual=[2.0],
        reduced_costs=[0.0],
        objective=2.0,
    )
    seen = {}

    def run(args, **kwargs):
        seen.update(json.loads(kwargs["input"]))
        with np.load(args[-1], allow_pickle=False) as data:
            seen["q"] = data["q"].copy()
            assert data["A_indices"].dtype == np.int32
        return SimpleNamespace(returncode=0, stderr="", stdout=adapter.PREFIX + json.dumps(payload))

    monkeypatch.setattr(adapter.subprocess, "run", run)
    return payload, seen


def test_explicit_gpu_transport_verifies_duals_and_budget(worker):
    payload, seen = worker
    r = solve(lp(), backend="cuopt-gpu", budget=SolveBudget(wall_time_s=10, threads=2))
    assert r.validation.valid and r.optimality_evidence.independently_verified_optimal
    assert r.objective == 2
    assert 0 < seen["time_limit_s"] <= 10 and seen["threads"] == 2
    assert not r.raw_statistics["reuse_applied"]


def test_maximize_uses_minimization_duals_and_native_objective(worker):
    payload, seen = worker
    payload.update(x=[3.0], dual=[-2.0], objective=-6.0)
    r = solve(lp(objective_sense="maximize", objective_offset=7.0), backend=CuOptBackend())
    assert r.objective == 13 and r.optimality_evidence.independently_verified_optimal
    np.testing.assert_equal(seen["q"], [-2.0])
    payload["objective"] = -5.0
    r = solve(lp(objective_sense="maximize", objective_offset=7.0), backend=CuOptBackend())
    assert not r.validation.valid


@pytest.mark.parametrize(
    "change",
    [
        dict(protocol="wrong"),
        dict(gpu_count=0),
        dict(gpu_count=True),
        dict(solved_by="DualSimplex"),
        dict(x=[1, 2]),
        dict(x=[np.nan]),
        dict(dual=[1, 2]),
        dict(reduced_costs=[np.inf]),
        dict(x=None),
        dict(objective=None),
        dict(objective=np.nan),
    ],
)
def test_malformed_or_cpu_response_rejected(worker, change):
    worker[0].update(change)
    with pytest.raises(RuntimeError):
        CuOptBackend().solve(lp())


@pytest.mark.parametrize(
    "termination,status",
    [
        ("PrimalInfeasible", "infeasible"),
        ("DualInfeasible", "infeasible_or_unbounded"),
        ("PrimalOrDualInfeasible", "infeasible_or_unbounded"),
        ("SomethingNew", "solver_error"),
        ("UnboundedOrInfeasible", "infeasible_or_unbounded"),
        ("TimeLimit", "limit_feasible"),
        ("IterationLimit", "limit_feasible"),
    ],
)
def test_status_does_not_promote_rays_or_unknown_termination(worker, termination, status):
    worker[0]["termination"] = termination
    r = CuOptBackend().solve(lp())
    assert r.backend_status == status
    assert (r.x is not None) == (status == "limit_feasible")


def test_no_solution_and_wall_timeout(worker, monkeypatch):
    worker[0].update(termination="TimeLimit", x=None)
    assert CuOptBackend().solve(lp()).backend_status == "limit_no_solution"

    def timeout(*args, **kwargs):
        raise adapter.subprocess.TimeoutExpired("worker", 1.0)

    monkeypatch.setattr(adapter.subprocess, "run", timeout)
    r = CuOptBackend().solve(lp())
    assert r.x is None and r.backend_status == "limit_no_solution"
    r = CuOptBackend(time_limit_s=1e-12).solve(lp())
    assert r.x is None and r.backend_status == "limit_no_solution"


@pytest.mark.parametrize(
    "options",
    [
        dict(time_limit_s=0),
        dict(time_limit_s=True),
        dict(time_limit_s=np.inf),
        dict(tolerance=0),
        dict(tolerance=np.nan),
        dict(tolerance=True),
        dict(threads=0),
        dict(threads=True),
        dict(iteration_limit=1.5),
    ],
)
def test_invalid_configuration(options):
    with pytest.raises(ValueError):
        CuOptBackend(**options).solve(lp())


def test_unsupported_models_rejected():
    with pytest.raises(ValueError, match="integer"):
        CuOptBackend().solve(lp(domains=["integer"]))
    with pytest.raises(TypeError, match="LinearProblem"):
        CuOptBackend().solve(QuadraticProblem(lp(), [[1.0]]))


def test_missing_runtime_and_no_automatic_routing(monkeypatch):
    monkeypatch.setattr(CuOptBackend, "is_available", lambda _: False)
    with pytest.raises(BackendUnavailableError):
        CuOptBackend().solve(lp())
    from solverpilot.runtime.auto import default_registry
    from solverpilot.runtime.catalog import backend_catalog

    assert "cuopt-gpu" not in default_registry().names()
    assert "cuopt-gpu" in backend_catalog()
    assert CuOptBackend().manifest.metadata["automatic_selection"] is False


@pytest.mark.parametrize("output", ["", adapter.PREFIX + "{}\n" + adapter.PREFIX + "{}"])
def test_invalid_protocol_envelope(worker, monkeypatch, output):
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stderr="", stdout=output),
    )
    with pytest.raises(RuntimeError):
        CuOptBackend().solve(lp())


def test_crashed_worker_is_unavailable(worker, monkeypatch):
    monkeypatch.setattr(
        adapter.subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=1, stderr="CUDA error", stdout=""),
    )
    with pytest.raises(BackendUnavailableError, match="CUDA error"):
        CuOptBackend().solve(lp())
