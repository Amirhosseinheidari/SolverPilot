"""Public highspy LP reuse, with native scenarios isolated from suite schedulers."""

import importlib.util
import os
import subprocess
import sys
import threading
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pytest

from solverpilot import LinearProblem, QuadraticProblem, execute
from solverpilot.backends import HighspyNativeBackend
from solverpilot.capabilities import Capability, SupportLevel
from solverpilot.plan import SolveBudget
from solverpilot.runtime.budgeting import apply_budget
from solverpilot.runtime.manifest import backend_configuration
from solverpilot.runtime.options import SolveOptions
from solverpilot.runtime.unified import solve_any


def lp(**changes):
    data = {"A": [[1., 1.]], "c": [1., 2.], "variable_lower": [0., 0.],
            "variable_upper": [10., 10.], "constraint_lower": [1.],
            "constraint_upper": [np.inf]}
    data.update(changes)
    return LinearProblem.from_data(**data)


def backend(**changes):
    settings = {"reuse": True, "solver": "simplex", "presolve": False, "threads": 1}
    settings.update(changes)
    return HighspyNativeBackend(**settings)


class NativeSpy:
    def __init__(self, factory):
        self.factory = factory
        self.native = factory.real()
        self.calls = Counter()
        self.failures = dict(factory.next_failures)
        factory.next_failures.clear()
        self.block = None

    def __getattr__(self, name):
        target = getattr(self.native, name)
        if not name.startswith("change") and name not in {"passModel", "run", "setOptionValue"}:
            return target

        def invoke(*args):
            self.calls[name] += 1
            if name in self.failures:
                failure = self.failures.pop(name)
                if isinstance(failure, Exception):
                    raise failure
                return failure
            if name != "run":
                return target(*args)
            with self.factory.lock:
                self.factory.active += 1
                self.factory.max_active = max(self.factory.max_active, self.factory.active)
            try:
                if self.block is not None:
                    entered, release = self.block
                    entered.set()
                    assert release.wait(5), "test did not release native run"
                    self.block = None
                if self.factory.delay:
                    time.sleep(self.factory.delay)
                return target(*args)
            finally:
                with self.factory.lock:
                    self.factory.active -= 1
        return invoke


@contextmanager
def native_spies():
    import highspy

    class Factory:
        real = highspy.Highs

        def __init__(self):
            self.created = []
            self.next_failures = {}
            self.lock = threading.Lock()
            self.active = self.max_active = 0
            self.delay = 0

        def __call__(self):
            spy = NativeSpy(self)
            self.created.append(spy)
            return spy

    factory = Factory()
    with patch.object(highspy, "Highs", factory):
        yield factory


def assert_equivalent(problem, warm):
    cold = execute(problem, backend(reuse=False))
    assert warm.status == cold.status
    assert warm.objective == pytest.approx(cold.objective) if cold.objective is not None else warm.objective is None
    if warm.x is not None:
        assert warm.validation.valid and cold.validation.valid


def scenario_updates():
    with native_spies() as spies:
        b = backend()
        p = lp()
        first = execute(p, b)
        original_x = first.x.copy()
        q = lp(c=[2, 1], variable_lower=[1, 0], variable_upper=[3, 4],
               constraint_lower=[2], objective_offset=-3)
        second = execute(q, b)
        assert first.objective == pytest.approx(1)
        assert second.objective == pytest.approx(0)
        assert np.array_equal(first.x, original_x)
        assert len(spies.created) == 1
        native = spies.created[0]
        assert native.calls["passModel"] == 1
        assert all(native.calls[name] == 1 for name in (
            "changeColsCost", "changeColsBounds", "changeRowsBounds", "changeObjectiveOffset"))
        raw = second.raw_statistics
        assert raw["reuse_applied"] and raw["reuse_mode"] == "native_lp_update"
        assert raw["updated_fields"] == ("costs", "column_bounds", "row_bounds", "objective_offset")
        observations = raw["reuse_observations"]
        assert observations["basis_valid_before_update"] == observations["basis_valid_before_run"] == True
        assert observations["basis_valid_after_run"] == bool(native.getBasis().valid)
        assert observations["factorization_reuse"] == "not_observed"
        assert raw["phase_timings"]["backend_build_s"] == 0
        assert raw["phase_timings"]["backend_update_s"] >= 0
        third = execute(q, b)
        assert third.raw_statistics["reuse_mode"] == "native_lp_resolve"
        assert len(spies.created) == 1 and native.calls["passModel"] == 1
        assert_equivalent(q, second)


def scenario_maximum_and_offset():
    b = backend()
    p = lp(c=[3, 2], constraint_lower=[-np.inf], constraint_upper=[4],
           objective_sense="maximize", objective_offset=7)
    assert execute(p, b).objective == pytest.approx(19)
    q = replace(p, c=[2, 3], constraint_upper=[5], variable_lower=[0, 1],
                variable_upper=[3, 4], objective_offset=-3)
    result = execute(q, b)
    assert result.objective == pytest.approx(11)
    assert result.trace.reuse_applied
    assert_equivalent(q, result)
    changed = execute(replace(q, objective_sense="minimize"), b)
    assert not changed.trace.reuse_applied
    assert changed.raw_statistics["reuse_reason"] == "objective_sense_changed"


def scenario_infeasible_and_unbounded():
    b = backend()
    execute(lp(), b)
    infeasible = execute(lp(constraint_lower=[25]), b)
    assert infeasible.backend_status == "infeasible"
    assert infeasible.x is None and infeasible.objective is None
    assert infeasible.trace.reuse_applied
    recovered = execute(lp(c=[2, 1], constraint_lower=[2]), b)
    assert recovered.objective == pytest.approx(2) and recovered.trace.reuse_applied
    assert_equivalent(lp(c=[2, 1], constraint_lower=[2]), recovered)
    p = lp(A=[[1, 0]], c=[0, -1], constraint_lower=[-np.inf],
           constraint_upper=[1], variable_upper=[np.inf, np.inf])
    assert execute(p, b).backend_status == "unbounded"
    q = replace(p, variable_upper=[np.inf, 3])
    bounded = execute(q, b)
    assert bounded.objective == pytest.approx(-3) and bounded.trace.reuse_applied


def scenario_rebuild_boundaries():
    with native_spies() as spies:
        b = backend()
        execute(lp(), b)
        changed = execute(lp(A=[[2, 1]]), b)
        assert changed.raw_statistics["reuse_reason"] == "matrix_values_changed"
        assert not changed.trace.reuse_applied and changed.objective == pytest.approx(.5)
        structural = lp(A=[[1, 1], [1, 0]], constraint_lower=[1, -np.inf], constraint_upper=[np.inf, 2])
        assert execute(structural, b).raw_statistics["reuse_reason"] == "structure_or_domains_changed"
        assert len(spies.created) == 3
        mip = lp(domains=["integer", "continuous"])
        result = execute(mip, b)
        assert result.validation.valid and not result.trace.reuse_applied and b._highs is None
        assert result.raw_statistics["reuse_reason"] == "nonreusable_problem"
        qp = QuadraticProblem.from_data(P=[[2]], A=[[1]], q=[-2], variable_lower=[0],
            variable_upper=[3], constraint_lower=[-np.inf], constraint_upper=[np.inf])
        result = execute(qp, b)
        assert result.x[0] == pytest.approx(1, abs=1e-6)
        assert not result.trace.reuse_applied and b._highs is None


def scenario_configuration():
    with native_spies() as spies:
        b = backend()
        execute(lp(), b)
        b.time_limit_s = 2
        assert execute(lp(), b).trace.reuse_applied
        b.presolve = True
        assert execute(lp(), b).raw_statistics["reuse_reason"] == "configuration_changed"
        assert len(spies.created) == 2
        b.solver = "ipm"
        result = execute(lp(), b)
        assert result.objective == pytest.approx(1)
        assert not result.trace.reuse_applied and b._highs is None
        b.solver = "invalid_solver"
        with pytest.raises(RuntimeError, match="option solver"):
            execute(lp(), b)
        assert b._highs is None


def scenario_update_recovery():
    import highspy

    for fault in (highspy.HighsStatus.kError, highspy.HighsStatus.kWarning, RuntimeError("native update raised")):
        with native_spies() as spies:
            b = backend()
            execute(lp(), b)
            spies.created[0].failures["changeRowsBounds"] = fault
            q = lp(c=[10, 1], constraint_lower=[2])
            result = execute(q, b)
            assert result.objective == pytest.approx(2)
            assert len(spies.created) == 2
            assert spies.created[0].calls["changeColsCost"] == 1
            assert spies.created[1].calls["passModel"] == 1
            assert not result.trace.reuse_applied
            assert result.raw_statistics["reuse_reason"] == "native_update_failed"
            assert result.raw_statistics["reuse_fallback_error"]
            assert result.raw_statistics["phase_timings"]["backend_update_s"] > 0
            assert result.raw_statistics["phase_timings"]["backend_build_s"] > 0
            assert execute(q, b).trace.reuse_applied


def scenario_failed_run_and_failed_rebuild():
    import highspy

    with native_spies() as spies:
        b = backend()
        execute(lp(), b)
        # A failing run leaves the old native OPTIMAL status untouched: it must
        # not be returned as a fresh optimal result.
        spies.created[0].failures["run"] = highspy.HighsStatus.kError
        with pytest.raises(RuntimeError, match="run failed"):
            execute(lp(), b)
        assert b._highs is None
        result = execute(lp(), b)
        assert not result.trace.reuse_applied and result.objective == pytest.approx(1)
        spies.created[-1].failures["changeRowsBounds"] = highspy.HighsStatus.kError
        spies.next_failures["passModel"] = highspy.HighsStatus.kError
        with pytest.raises(RuntimeError, match="rejected model"):
            execute(lp(constraint_lower=[2]), b)
        assert b._highs is None
        assert execute(lp(constraint_lower=[3]), b).objective == pytest.approx(3)


def scenario_budget_and_cloning():
    b = backend()
    budget = SolveBudget(wall_time_s=2, threads=1)
    first = execute(lp(), apply_budget(b, budget))
    native = b._highs
    second = execute(lp(c=[2, 1], constraint_lower=[2]), apply_budget(b, budget))
    assert b._highs is native and second.trace.reuse_applied
    assert b.time_limit_s is None  # call-local budget restored
    assert second.trace.parameters["backend_configuration"]["reuse"] is True
    raw = second.raw_statistics
    assert raw["native_runtime_before_s"] >= first.raw_statistics["native_runtime_after_s"]
    assert raw["native_time_limit_s"] == pytest.approx(raw["native_runtime_before_s"] + 2)
    assert native.getOptionValue("time_limit")[1] == raw["native_time_limit_s"]
    clone = replace(b, time_limit_s=3)
    assert clone._highs is None and clone._lock is not b._lock
    assert not execute(lp(), clone).trace.reuse_applied
    clone.close()
    assert b._highs is native and not b._closed


def scenario_changing_call_budgets():
    with native_spies() as spies:
        b = backend()
        native = None
        # Exercise the common API's setup-time subtraction as well as growing,
        # shrinking and removing a finite allowance on the same native model.
        for index, allowance in enumerate((2., 3., .5, None, 1., None)):
            options = SolveOptions(budget=None if allowance is None else SolveBudget(wall_time_s=allowance))
            _, result = solve_any(lp(constraint_lower=[index + 1]), backend=b, options=options)
            assert result.objective == pytest.approx(index + 1)
            assert result.validation.valid
            assert result.trace.reuse_applied == (index > 0)
            assert b.time_limit_s is None
            if native is None:
                native = b._highs
            assert b._highs is native
            raw = result.raw_statistics
            configured_limit = native.getOptionValue("time_limit")[1]
            if allowance is None:
                assert raw["native_time_limit_s"] is None
                assert raw["requested_time_limit_s"] is None
                assert np.isinf(configured_limit)
            else:
                assert 0 < raw["requested_time_limit_s"] <= allowance
                assert raw["native_time_limit_s"] == pytest.approx(
                    raw["native_runtime_before_s"] + raw["requested_time_limit_s"])
                assert configured_limit == raw["native_time_limit_s"]
        assert len(spies.created) == 1
        assert native.calls["passModel"] == 1
        b.time_limit_s = 0
        with pytest.raises(ValueError, match="finite and positive"):
            b.solve(lp())
        assert b._highs is None


def scenario_defaults_and_reset():
    with native_spies() as spies:
        for settings in ({"reuse": False}, {"solver": None}):
            b = backend(**settings)
            start = len(spies.created)
            for _ in range(2):
                result = execute(lp(), b)
                assert not result.trace.reuse_applied and b._highs is None
                assert result.trace.reuse_mode == "cold_native_solve"
            assert len(spies.created) == start + 2
        b = backend()
        execute(lp(), b)
        b.reset()
        assert b._highs is None
        result = execute(lp(), b)
        assert not result.trace.reuse_applied
        assert result.raw_statistics["reuse_reason"] == "explicit_reset"
        with b:
            assert execute(lp(), b).trace.reuse_applied
        assert b._closed and b._highs is None
        b.close()  # idempotent, but never reopens
        with pytest.raises(RuntimeError, match="closed"):
            b.solve(lp())
        with pytest.raises(RuntimeError, match="closed"):
            b.reset()


def scenario_concurrency_and_close():
    with native_spies() as spies:
        spies.delay = .003
        b = backend()
        problems = [lp(c=[2, 1], constraint_lower=[i / 4]) for i in range(1, 9)]
        with ThreadPoolExecutor(max_workers=4) as pool:
            results = list(pool.map(b.solve, problems))
        assert spies.max_active == 1 and len(spies.created) == 1
        assert [r.objective_reported for r in results] == pytest.approx([i / 4 for i in range(1, 9)])
        entered, release, close_started = threading.Event(), threading.Event(), threading.Event()
        spies.created[0].block = (entered, release)

        def close():
            close_started.set()
            b.close()

        with ThreadPoolExecutor(max_workers=2) as pool:
            solving = pool.submit(b.solve, lp())
            assert entered.wait(5)
            closing = pool.submit(close)
            assert close_started.wait(5)
            assert not closing.done()
            release.set()
            assert solving.result(timeout=5).objective_reported == pytest.approx(1)
            closing.result(timeout=5)
        assert b._closed and b._highs is None


SCENARIOS = ("updates", "maximum_and_offset", "infeasible_and_unbounded", "rebuild_boundaries",
             "configuration", "update_recovery", "failed_run_and_failed_rebuild",
             "budget_and_cloning", "changing_call_budgets", "defaults_and_reset", "concurrency_and_close")


@pytest.mark.native
@pytest.mark.parametrize("scenario", SCENARIOS)
def test_native_reuse_in_fresh_process(scenario):
    if importlib.util.find_spec("highspy") is None:
        pytest.skip("optional highspy not installed")
    path = Path(__file__).resolve()
    # One child per scenario avoids process-global thread-count/ABI interference
    # with other optional native backends in the full suite.
    code = f"import runpy; m=runpy.run_path({str(path)!r}); m['scenario_{scenario}']()"
    env = dict(os.environ)
    env["PYTHONPATH"] = str(path.parents[1] / "src") + os.pathsep + env.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True,
                            text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_manifest_and_configuration_do_not_claim_general_same_sparsity_updates():
    with patch.object(HighspyNativeBackend, "is_available", return_value=False):
        cold = backend(reuse=False).manifest
        warm = backend().manifest
    assert cold.capabilities[Capability.NATIVE_REOPTIMIZATION] is SupportLevel.UNSUPPORTED
    assert warm.capabilities[Capability.NATIVE_REOPTIMIZATION] is SupportLevel.NATIVE
    assert warm.capabilities[Capability.SAME_SPARSITY_DATA_UPDATE] is SupportLevel.UNKNOWN
    assert warm.capabilities[Capability.BASIS_START] is SupportLevel.UNKNOWN
    assert warm.metadata["matrix_update_policy"] == "rebuild even when only matrix values change"
    assert backend_configuration(backend()) == {"time_limit_s": None, "threads": 1,
        "presolve": False, "solver": "simplex", "reuse": True}
    with pytest.raises(TypeError, match="boolean"):
        HighspyNativeBackend(reuse="yes")
