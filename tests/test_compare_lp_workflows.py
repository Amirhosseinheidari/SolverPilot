import importlib.util
import os
import subprocess
import sys
from pathlib import Path
from time import perf_counter

import numpy as np
import pytest

PATH = Path(__file__).resolve().parents[1] / "benchmarks" / "compare_lp_workflows.py"
SPEC = importlib.util.spec_from_file_location("compare_lp_workflows", PATH)
bench = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = bench
SPEC.loader.exec_module(bench)


@pytest.mark.parametrize("family", ["production", "network"])
@pytest.mark.parametrize("power", [0, 16])
def test_generator_has_independent_analytic_witnesses_and_real_updates(family, power):
    sequence = bench.make_sequence(family, 16, power, 124, 4)
    for case in sequence:
        assert bench.independent_check(case, case.optimum_x, case.reference_objective)["passed"]
        if family == "production":
            assert np.all(case.dual_witness >= 0)
            np.testing.assert_allclose(case.c, -case.A.T @ case.dual_witness)
        else:
            reduced = case.c - case.A.T @ case.dual_witness
            assert np.all(reduced >= 0)
            assert float(reduced @ case.optimum_x) == pytest.approx(0)
        assert not case.c.flags.writeable
        assert not case.A.data.flags.writeable
    assert np.array_equal(sequence[0].c, sequence[1].c)
    assert not np.array_equal(sequence[0].row_upper, sequence[1].row_upper)
    assert not np.array_equal(sequence[1].c, sequence[2].c)
    assert np.array_equal(sequence[1].row_upper, sequence[2].row_upper)
    assert bench.digest([c.payload() for c in sequence]) == bench.digest(
        [c.payload() for c in bench.make_sequence(family, 16, power, 124, 4)])


def test_checker_rejects_missing_nonfinite_infeasible_and_feasible_suboptimal_candidates():
    case = bench.make_sequence("production", 8, 0, 1, 2)[0]
    for x, objective in ((None, None), (np.full(8, np.nan), 0),
                         (case.upper * 100, case.reference_objective),
                         (np.zeros(8), 0), (case.optimum_x, case.reference_objective + 1)):
        assert not bench.independent_check(case, x, objective)["passed"]
    check = bench.independent_check(case, np.zeros(8), 0)
    assert check["primal_feasible"] and check["objective_consistent"]
    assert not check["reference_agrees"]


def test_counterbalancing_each_workflow_occupies_each_position_and_pair_reverses():
    names = bench.WORKFLOWS
    orders = [bench.counterbalanced_order(names, r) for r in range(8)]
    for name in names:
        assert [o.index(name) for o in orders].count(0) == 2
        assert {o.index(name) for o in orders} == set(range(4))
    assert bench.counterbalanced_order(names[:2], 0) == names[:2]
    assert bench.counterbalanced_order(names[:2], 1) == names[:2][::-1]


class OracleAdapter:
    reuse = "test retained state"
    constructions = updates = solves = 0

    def __init__(self, case, settings):
        type(self).constructions += 1
        self.case = case

    def update(self, case):
        type(self).updates += 1
        self.case = case

    def solve(self, warm):
        type(self).solves += 1
        return {"status": "optimal", "optimal": True, "x": self.case.optimum_x,
                "objective": self.case.reference_objective}


def test_cold_and_warm_measure_real_construction_update_counts():
    sequence = bench.make_sequence("production", 8, 0, 1, 4)
    for mode, constructions, updates in (("cold", 4, 0), ("warm_sequence", 1, 3)):
        OracleAdapter.constructions = OracleAdapter.updates = OracleAdapter.solves = 0
        rows = bench.run_sequence(OracleAdapter, sequence, mode, {}, perf_counter() + 10)
        assert all(row["accepted"] for row in rows)
        assert (OracleAdapter.constructions, OracleAdapter.updates, OracleAdapter.solves) == (constructions, updates, 4)
        for row in rows:
            assert row["checked_total_s"] >= row["workflow_s"] >= 0
            assert row["workflow_s"] == pytest.approx(row["construction_or_update_s"] + row["solve_api_s"])


def test_failure_is_counted_and_excluded_from_successful_timing_statistics():
    class Incorrect(OracleAdapter):
        def solve(self, warm):
            out = super().solve(warm)
            out["x"] = np.zeros_like(self.case.optimum_x)
            out["objective"] = 0
            return out

    config = bench.Config(preset="smoke", steps=2, repeats=2, warmups=1,
                          workflows=("solverpilot", "direct-highspy"))
    result = bench.run_benchmark(config, factories={"solverpilot": OracleAdapter,
        "direct-highspy": Incorrect}, availability={"solverpilot": {"available": True},
                                                    "direct-highspy": {"available": True}})
    assert not result["requested_matrix_complete"]
    assert result["actual_measured_rows"] == result["expected_measured_rows"]
    bad = [row for row in result["summary"] if row["workflow"] == "direct-highspy"]
    assert bad and all(row["failures"] == row["attempts"] for row in bad)
    assert all(row["checked_total_s"]["count"] == 0 for row in bad)
    assert all(row["checked_total_s"]["median"] is None for row in bad)
    assert sum(r["attempts"] for r in result["summary"]) == result["actual_measured_rows"]


def test_missing_workflow_and_budget_exhaustion_cannot_be_complete():
    config = bench.Config(preset="smoke", steps=2, repeats=1, warmups=0,
                          workflows=("solverpilot", "direct-highspy"))
    result = bench.run_benchmark(config, factories={"solverpilot": OracleAdapter},
        availability={"solverpilot": {"available": True},
                      "direct-highspy": {"available": False, "reason": "test missing"}})
    assert not result["requested_matrix_complete"]
    assert result["actual_measured_rows"] * 2 == result["expected_measured_rows"]
    rows = bench.run_sequence(OracleAdapter, bench.make_sequence("network", 8, 0, 1, 2),
                              "cold", {}, perf_counter() - 1)
    assert all(r["state"] == "budget_exhausted" and not r["accepted"] for r in rows)


def test_failed_update_discards_adapter_before_next_step():
    class FailsOnce(OracleAdapter):
        def update(self, case):
            if case.step == 1:
                raise RuntimeError("failed update")
            super().update(case)

    rows = bench.run_sequence(FailsOnce, bench.make_sequence("production", 8, 0, 1, 4),
                              "warm_sequence", {}, perf_counter() + 10)
    assert rows[1]["state"] == "error"
    assert rows[2]["accepted"] and rows[2]["fresh_model"]


@pytest.mark.parametrize("kwargs", [{"steps": 1}, {"repeats": 0}, {"seed": -1},
    {"max_wall_s": float("nan")}, {"workflows": ("solverpilot", "solverpilot")}])
def test_config_rejects_invalid_or_unbounded_protocol(kwargs):
    with pytest.raises(ValueError):
        bench.Config(**kwargs)


@pytest.mark.native
def test_available_real_adapters_pass_shared_checker_for_cold_and_updates():
    if importlib.util.find_spec("highspy") is None:
        pytest.skip("optional highspy not installed")
    # Native HiGHS owns a process-global scheduler. Other suite tests may have
    # initialized it with different thread counts; this check gets its own process.
    code = f"""
import runpy
from time import perf_counter
b = runpy.run_path({str(PATH)!r})
config = b['Config'](preset='smoke', steps=3, repeats=1, warmups=0)
factories, availability, settings, native = b['discover'](config)
assert native['version']
assert {{'solverpilot', 'direct-highspy'}} <= factories.keys()
assert settings['threads'] == 1
if 'cvxpy-highs' in factories:
    assert len(availability['cvxpy-highs']['canonicalization_probes']) == 2
for family in ('production', 'network'):
    sequence = b['make_sequence'](family, 8, 16, 23, 3)
    for name, factory in factories.items():
        if name == 'cvxpy-highs':
            adapter = factory(sequence[0], settings)
            for case in sequence:
                adapter.update(case)
                probe = b['cvxpy_formulation_probe'](adapter, case)
                assert probe['native_shape'] == list(case.A.shape)
                assert probe['exact_data_match'] and probe['native_column_bounds']
            # The former explicit-bound formulation must fail the fair-matrix guard.
            cp, case = adapter.cp, sequence[0]
            x = cp.Variable(len(case.c))
            resource = case.A @ x <= case.row_upper if family == 'production' else case.A @ x == case.row_upper
            adapter.model = cp.Problem(cp.Minimize(case.c @ x),
                [x >= case.lower, x <= case.upper, resource])
            try:
                b['cvxpy_formulation_probe'](adapter, case)
            except ValueError:
                pass
            else:
                raise AssertionError('expanded bound rows must not pass the formulation probe')
        for mode in ('cold', 'warm_sequence'):
            rows = b['run_sequence'](factory, sequence, mode, settings, perf_counter() + 20)
            assert all(row['accepted'] for row in rows), (name, rows)
for name in set(b['WORKFLOWS']) - factories.keys():
    assert not availability[name]['available'] and availability[name]['reason']
"""
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(PATH.parents[1] / "src") + os.pathsep + environment.get("PYTHONPATH", "")
    result = subprocess.run([sys.executable, "-c", code], env=environment,
                            capture_output=True, text=True, timeout=60, check=False)
    assert result.returncode == 0, result.stdout + result.stderr


def test_cli_persists_incomplete_matrix_and_refuses_overwrite(tmp_path, monkeypatch):
    monkeypatch.setattr(bench, "discover", lambda config: (
        {"solverpilot": OracleAdapter},
        {name: {"available": name == "solverpilot"} for name in config.workflows}, {}, {}))
    path = tmp_path / "comparison.json"
    args = ["--output", str(path), "--preset", "smoke", "--steps", "2",
            "--repeats", "1", "--warmups", "0"]
    assert bench.main(args) == 2
    payload = bench.json.loads(path.read_text())
    assert payload["observations"] and not payload["requested_matrix_complete"]
    previous = path.read_bytes()
    with pytest.raises(SystemExit):
        bench.main(args)
    assert path.read_bytes() == previous
