import numpy as np
import pytest
from scipy import sparse
from solverpilot import Model, QuadraticProblem, LinearProblem, execute
from solverpilot.backends import OSQPNativeBackend, ScipyHighsLPBackend
from solverpilot.analysis import (
    shadow_prices,
    parameter_sensitivity,
    differentiate_qp,
    quality_report,
    scaling_report,
)
from solverpilot.scenarios import scenario_sweep, scenario_statistics
from solverpilot.applications import (
    production_model,
    transportation_model,
    energy_dispatch_model,
    mpc_model,
)
from solverpilot.model.robust import robust_leq, minimize_worst_case
from solverpilot.runtime.unified import solve_any
from solverpilot.runtime.options import SolveOptions


def qp(q=-4.0, lower=-10.0, upper=10.0):
    return QuadraticProblem.from_data(
        P=[[2.0]],
        q=[q],
        A=sparse.csr_matrix((0, 1)),
        variable_lower=[lower],
        variable_upper=[upper],
        constraint_lower=[],
        constraint_upper=[],
    )


def test_named_dual_marginal_matches_capacity_perturbation():
    model = production_model([3.0, 2.0], [[1.0, 1.0]], [4.0])
    compiled = model.compile()
    result = compiled.solve(backend=ScipyHighsLPBackend())
    prices = shadow_prices(model, compiled, result)
    assert prices[0].name == "resource_capacity"
    assert prices[0].marginal_value == pytest.approx(-3.0)  # canonical row is -x >= -capacity
    parameter_price = parameter_sensitivity(model, compiled, result, "capacity")["marginal"][0]
    assert parameter_price == pytest.approx(3.0)
    assert (
        "independent_numerical_bound"
        == quality_report(compiled.execution_ir, result)["summary"]["optimality"]
    )
    other = production_model([3.0, 2.0], [[1.0, 1.0]], [5.0]).compile().solve()
    assert other.objective - result.objective == pytest.approx(parameter_price)


def test_qp_derivative_matches_finite_difference():
    pytest.importorskip("osqp")
    backend = OSQPNativeBackend(eps_abs=1e-10, eps_rel=1e-10)
    p = qp()
    result = execute(p, backend)
    derivative = differentiate_qp(p, result, dq=[1.0])
    epsilon = 1e-4
    forward, backward = execute(qp(-4.0 + epsilon), backend), execute(qp(-4.0 - epsilon), backend)
    assert derivative.dx == pytest.approx((forward.x - backward.x) / (2 * epsilon), abs=1e-6)
    assert derivative.objective_derivative == pytest.approx(
        (forward.objective - backward.objective) / (2 * epsilon), abs=1e-6
    )
    assert derivative.dx[0] == pytest.approx(-0.5)


def test_qp_derivative_rejects_kink_and_handles_active_bound():
    pytest.importorskip("osqp")
    backend = OSQPNativeBackend(eps_abs=1e-10, eps_rel=1e-10)
    p = qp(-4.0, upper=1.0)
    derivative = differentiate_qp(p, execute(p, backend), du=[1.0])
    assert derivative.dx[0] == pytest.approx(1.0)
    assert derivative.objective_derivative == pytest.approx(-2.0)
    p = qp(-2.0, upper=1.0)
    with pytest.raises(ValueError, match="degenerate"):
        differentiate_qp(p, execute(p, backend), dq=[1.0])


def test_free_qp_strong_convexity_bounds_stationarity_error():
    from solverpilot.validate import verify_optimality

    p = qp(lower=-np.inf, upper=np.inf)
    assert verify_optimality(p, [2.0 + 1e-10], [0.0]).verified
    assert not verify_optimality(p, [3.0], [0.0]).verified


def test_scenarios_reset_omitted_values_and_do_not_mutate_source():
    model = production_model([3.0, 2.0], [[1.0, 1.0]], [4.0])
    before = model.data_hash
    rows = list(
        scenario_sweep(
            model,
            [("high", {"capacity": [5.0]}), ("base", {}), ("bad", {"unknown": 2}), ("base2", {})],
            on_error="record",
        )
    )
    assert [r.summary.objective if r.summary else None for r in rows] == [15.0, 12.0, None, 12.0]
    assert model.data_hash == before
    assert scenario_statistics(rows)["errors"] == 1


def test_box_robust_constraint_matches_all_vertices():
    model = Model()
    x = model.variable(2, lower=0, upper=10)
    robust_leq(model, [1.0, 2.0], [0.5, 0.25], x, 6.0)
    model.maximize(x.sum())
    compiled = model.compile()
    result = compiled.solve()
    original = compiled.reconstruct_primal(result.x)
    assert result.validation.valid
    for a in (0.5, 1.5):
        for b in (1.75, 2.25):
            assert a * original[0] + b * original[1] <= 6.0 + 1e-6
    assert original.sum() == pytest.approx(4.0)


def test_minimax_objective_analytic_balancing_solution():
    model = Model()
    x = model.variable(2, lower=0, upper=1)
    model.add(x.sum() == 1)
    minimize_worst_case(model, x, [[1.0, 0.0], [0.0, 1.0]])
    result = model.compile().solve()
    assert result.validation.valid and result.objective == pytest.approx(0.5)


@pytest.mark.parametrize(
    "factory, expected",
    [
        (lambda: transportation_model([[1.0, 5.0], [4.0, 2.0]], [3.0, 4.0], [3.0, 4.0]), 11.0),
        (lambda: energy_dispatch_model([3.0, 7.0], [1.0, 4.0], [5.0, 5.0]), 16.0),
        (lambda: mpc_model([[1.0]], [[1.0]], [[1.0]], [[1.0]], 1, [1.0]), 1.5),
    ],
)
def test_application_templates_analytic_objectives(factory, expected):
    model = factory()
    summary, result = solve_any(model)
    assert summary.feasible and summary.objective == pytest.approx(expected, abs=1e-6)
    assert model.compile().validate_original(model, result.x, atol=1e-5).valid


def test_unified_nlp_and_cp_and_explicit_unsupported_controls():
    from solverpilot.cp import CPModel

    model = CPModel()
    x = model.int_var(0, 2, name="x")
    model.minimize(x)
    summary, result = solve_any(model)
    assert summary.feasible and summary.objective == 0
    pytest.importorskip("casadi")
    m = Model()
    x = m.variable(lower=-1.0, upper=1.0)
    m.minimize(x.exp())
    from solverpilot import SolveBudget

    summary, result = solve_any(m, options=SolveOptions(budget=SolveBudget(wall_time_s=10)))
    assert summary.feasible and summary.optimality == "local_candidate"
    assert summary.problem_data_hash == m.compile().execution_ir.data_hash


def test_scaling_flags_counterexample_model():
    p = LinearProblem.from_data(
        A=[[1e-10]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[1e10],
        constraint_lower=[0.0],
        constraint_upper=[1.0],
    )
    report = scaling_report(p)
    assert len(report["warnings"]) == 3


def test_cli_lists_and_verifies_clarabel(capsys):
    pytest.importorskip("clarabel")
    import json
    from solverpilot.cli.capabilities import main

    assert main(["--backend", "clarabel-native", "--verify"]) == 0
    payload = json.loads(capsys.readouterr().out)
    assert payload[0]["conformance_passed"]
