from dataclasses import replace
from fractions import Fraction
from itertools import product
from types import SimpleNamespace
import numpy as np
import pytest
from solverpilot.model import Model
from solverpilot.globalopt import (
    GlobalQuadraticProblem,
    FactorableProblem,
    SCIPGlobalBackend,
    solve_global,
)
from solverpilot.globalopt.intervals import Range, upper_float
from solverpilot.intelligence.shadow import observe_shadow_policy
from solverpilot.runtime.reuse import reuse_evidence
from solverpilot.runtime.integrations import integration_readiness
from solverpilot import LinearProblem, solve, SolveBudget


def lp():
    return LinearProblem.from_data(
        A=np.empty((0, 1)),
        c=[1],
        variable_lower=[0],
        variable_upper=[2],
        constraint_lower=[],
        constraint_upper=[],
    )


def test_shadow_observation_never_changes_production():
    p = lp()

    def selector(record):
        assert record.values["problem_class"] == "lp"
        with pytest.raises(TypeError):
            record.values["n_variables"] = 99
        return "alternate"

    args = dict(
        policy_id="heldout-experiment-1",
        production_backend="baseline",
        eligible_backends=["baseline", "alternate"],
    )
    r = observe_shadow_policy(p, selector=selector, **args)
    assert r.eligible and r.suggested_backend == "alternate"
    assert r.production_backend == "baseline" and not r.changes_production_routing
    assert r.overhead_s >= 0 and r.problem_data_hash == p.data_hash
    assert not observe_shadow_policy(p, selector=lambda _: "missing", **args).eligible

    def broken(_):
        raise RuntimeError("bad predictor")

    assert "RuntimeError" in observe_shadow_policy(p, selector=broken, **args).reason
    with pytest.raises(ValueError):
        observe_shadow_policy(p, selector=selector, **{**args, "eligible_backends": []})


def test_probe_installation_is_not_qualification(monkeypatch):
    monkeypatch.setattr(
        SCIPGlobalBackend, "capabilities", lambda _: dict(exact_milp=True, certificate_output=True)
    )
    exact, gpu = integration_readiness()
    assert exact.available and not exact.qualified
    assert not gpu.qualified and not gpu.details["gpu_execution_tested"]
    with pytest.raises(TypeError):
        exact.details["exact_milp"] = False

    def broken(_):
        raise OSError("missing native library")

    monkeypatch.setattr(SCIPGlobalBackend, "capabilities", broken)
    assert not integration_readiness()[0].available


def test_reuse_does_not_infer_factorization_from_workspace():
    r = SimpleNamespace(
        raw_statistics={
            "reuse_report": {"workspace": "observed", "primal_dual_start": "automatic_enabled"}
        }
    )
    e = reuse_evidence(r)
    assert e.workspace == "observed" and e.numeric_factorization == "unknown"
    assert e.primal_dual_start == "automatic_enabled"
    assert reuse_evidence(object()).workspace == "unknown"


def test_recovery_invalid_budget_does_not_execute_backend():
    class NeverSolve:
        def solve(self, p):
            raise AssertionError("must fail before solving")

    for v in (True, 0, -1, np.nan, np.inf):
        with pytest.raises(ValueError, match="certificate_recovery"):
            solve(lp(), backend=NeverSolve(), certificate_recovery=v)


def test_global_direct_ir_is_frozen_and_rejects_malformed_layout():
    m = Model()
    x = m.variable(lower=-2, upper=2)
    a = m.parameter(value=2)
    m.minimize(a * x * x)
    p = m.compile(target="global").execution_ir
    with pytest.raises(ValueError):
        p.relaxation.variable_lower.flags.writeable = True
    with pytest.raises(ValueError):
        next(iter(p.relaxation.parameter_values.values())).flags.writeable = True
    key = next(iter(p.relaxation.variable_layout))
    for layout in ({}, {key: (2, ())}, {key: (0, ()), "duplicate": (0, ())}):
        with pytest.raises(ValueError, match="layout"):
            FactorableProblem(replace(p.relaxation, variable_layout=layout), p.domains)
    with pytest.raises(ValueError, match="parameter"):
        FactorableProblem(
            replace(p.relaxation, parameter_values={a.id.value: np.array(np.nan)}), p.domains
        )
    node = replace(p.relaxation.objective_node, shape=(1,))
    with pytest.raises(ValueError, match="scalar"):
        FactorableProblem(replace(p.relaxation, objective_node=node), p.domains)


def test_exact_interval_encloses_rational_samples():
    a = Range(Fraction(-2), Fraction(3))
    b = Range(Fraction(1, 3), Fraction(2))
    for x in [Fraction(-2), Fraction(-1, 3), Fraction(0), Fraction(3)]:
        for y in [Fraction(1, 3), Fraction(1), Fraction(2)]:
            for enclosure, value in [
                (a / b, x / y),
                (a * b, x * y),
                (a - b, x - y),
                (2 - a, 2 - x),
                (a**0, x**0),
                (a**3, x**3),
                (a**4, x**4),
                (abs(a), abs(x)),
            ]:
                assert enclosure.lo <= value <= enclosure.hi
    with pytest.raises(ValueError):
        a / a
    assert Fraction(upper_float(Fraction(1, 3))) >= Fraction(1, 3)


@pytest.mark.native
def test_miqp_against_exhaustive_reference():
    pytest.importorskip("pyscipopt")
    rng = np.random.default_rng(401)
    candidates = np.array([a for a in product(range(-2, 3), repeat=2) if sum(a) <= 2], dtype=float)
    b = SCIPGlobalBackend(time_limit_s=10)
    for i in range(24):
        raw = rng.integers(-3, 4, (2, 2))
        P = raw + raw.T
        q = rng.integers(-3, 4, 2)
        p = GlobalQuadraticProblem.from_data(
            P=P,
            A=[[1, 1]],
            q=q,
            variable_lower=[-2, -2],
            variable_upper=[2, 2],
            constraint_lower=[-np.inf],
            constraint_upper=[2],
            domains=["integer"] * 2,
            objective_sense="minimize" if i % 2 == 0 else "maximize",
            objective_offset=1.25,
        )
        values = [p.objective_value(x) for x in candidates]
        expected = min(values) if i % 2 == 0 else max(values)
        r = solve_global(p, backend=b)
        assert r.validation.valid and abs(r.objective - expected) < 1e-6
        assert not r.optimality_evidence.independently_verified_optimal


@pytest.mark.native
def test_global_options_callbacks_and_reuse_after_error():
    pytest.importorskip("pyscipopt")
    m = Model()
    x = m.variable(8, lower=-2, upper=2)
    m.minimize(((x * x - 1) ** 2).sum())
    b = SCIPGlobalBackend()
    stopped = solve_global(m, backend=b, cancellation=SimpleNamespace(cancelled=True))
    assert stopped.status == "cancelled" and stopped.x is None
    events = []
    r = solve_global(m, backend=b, progress=lambda e: events.append(e) or True)
    assert events and r.backend_status in {"userinterrupt", "optimal"}

    def fail(_):
        raise ValueError("callback error")

    with pytest.raises(RuntimeError, match="callback failed"):
        solve_global(m, backend=b, progress=fail)
    r = solve_global(m, backend=b, budget=SolveBudget(wall_time_s=10, threads=1, memory_mb=512))
    assert r.validation.valid
    assert b.time_limit_s == 60.0
    tiny = solve_global(m, backend=SCIPGlobalBackend(time_limit_s=1e-12))
    assert tiny.backend_status == "timelimit" and tiny.x is None


@pytest.mark.native
def test_global_catalog_has_scoped_runtime_evidence():
    pytest.importorskip("pyscipopt")
    from solverpilot.runtime.catalog import capability_manifest, verify_specialized_backend
    from solverpilot.capabilities.v2 import CapabilityKey, CapabilityStatus

    b = SCIPGlobalBackend()
    manifest = capability_manifest(b)
    assert manifest.claims[CapabilityKey.PROBLEM_MIQP].status is CapabilityStatus.RESTRICTED
    assert not manifest.metadata["independent_global_proof"]
    assert verify_specialized_backend(b)


def test_release_gates_exercise_new_optional_paths():
    from pathlib import Path
    root=Path(__file__).resolve().parents[1]
    runner=(root/'tools/release_cell_runner.py').read_text()
    for file in ('test_04_global.py','test_04_hardening.py','test_04_pdlp.py','test_04_generalized_power.py','test_04_conic_bounds.py'):
        assert file in runner
    assert 'expected 24 public examples' in runner
