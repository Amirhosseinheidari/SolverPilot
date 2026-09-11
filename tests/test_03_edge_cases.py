import json
import numpy as np
import pytest
from solverpilot import Model, LinearProblem, QuadraticProblem
from solverpilot.runtime import SolveOptions, solve_any, solve_verified, UnverifiedSolutionError
from solverpilot.runtime.batch import (
    CancellationToken,
    solve_batch,
    BatchExecutor,
    iter_solve_batch,
)
from solverpilot.runtime.manifest import save_run, replay_run
from solverpilot.model import abs, norm, huber, quad_form
from solverpilot.model.robust import robust_leq, minimize_worst_case
from solverpilot.applications import production_model
from solverpilot.scenarios import scenario_sweep


def lp():
    return LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )


def test_sequential_and_isolated_batch_agree_with_process_mode():
    outputs = [
        solve_batch([lp(), lp()], backend="scipy-highs-ds", mode=mode)
        for mode in ("sequential", "process", "isolated")
    ]
    assert all([r.objective for r in output] == [1.0, 1.0] for output in outputs)
    assert all(r.validation_valid for output in outputs for r in output)


def test_crashed_worker_is_reported_and_replaced():
    with BatchExecutor(backend="scipy-highs-ds", max_workers=1, timeout_s=30) as executor:
        assert list(executor.iter([lp()]))[0].validation_valid
        process = executor._workers[0].process
        process.terminate()
        process.join()
        assert list(executor.iter([lp()]))[0].status == "error"
        assert list(executor.iter([lp()]))[0].validation_valid
        assert executor._workers[0].process is not process


@pytest.mark.parametrize(
    "options",
    [
        {"max_workers": 0},
        {"max_pending": 0},
        {"max_workers": True},
        {"timeout_s": np.inf},
        {"backend": object()},
    ],
)
def test_invalid_batch_controls_fail_before_input_consumption(options):
    with pytest.raises((ValueError, TypeError)):
        list(iter_solve_batch([], **options))


def test_unsupported_controls_are_explicit():
    from solverpilot.exceptions import BudgetNotSupportedError

    with pytest.raises(BudgetNotSupportedError):
        list(iter_solve_batch([lp()], mode="sequential", timeout_s=1))
    with pytest.raises(BudgetNotSupportedError):
        solve_any(lp(), options=SolveOptions(cancellation=CancellationToken()))
    m = Model()
    x = m.variable()
    m.minimize(x)
    with pytest.raises(ValueError, match="twice"):
        m.compile().solve(options=SolveOptions(), tolerances=SolveOptions().tolerances)


def test_strict_solve_accepts_bound_and_retains_unverified_result():
    summary, result = solve_verified(lp(), backend="scipy-highs-ds")
    assert summary.feasible
    # SLSQP provides a feasible QP candidate but no canonical dual evidence.
    p = QuadraticProblem.from_data(
        P=[[2.0]],
        q=[-2.0],
        A=np.empty((0, 1)),
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    with pytest.raises(UnverifiedSolutionError) as error:
        solve_verified(p, backend="scipy-slsqp-qp-bridge")
    assert error.value.result.x is not None and error.value.summary.feasible


def test_qp_manifest_replay(tmp_path):
    pytest.importorskip("osqp")
    m = Model()
    x = m.variable(lower=0.0, upper=2.0)
    m.minimize((x - 1.0) ** 2)
    p = m.compile().execution_ir
    summary, result = solve_any(p, backend="osqp-native")
    path = tmp_path / "qp.json"
    save_run(path, p, result, include_model=True)
    assert replay_run(path).objective == pytest.approx(result.objective, abs=1e-7)
    payload = json.loads(path.read_text())
    payload["environment"]["packages"]["numpy"] = "0.0"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="versions"):
        replay_run(path)


def test_atoms_check_domains_and_objective_constraints():
    m = Model()
    x = m.variable(2)
    for call in (
        lambda: norm(x, 3),
        lambda: huber(x, -1),
        lambda: quad_form(x, [[1, 0], [0, -1]]),
        lambda: norm(x * x),
        lambda: robust_leq(m, [1, 1], [-1, 1], x, 1),
        lambda: minimize_worst_case(m, x, [[1]]),
    ):
        with pytest.raises(ValueError):
            call()
    m.minimize((m.constant([1, -1]) * abs(x)).sum())
    with pytest.raises(ValueError):
        m.compile()


def test_atom_analytic_values_free_variables_and_original_constraints():
    pytest.importorskip("clarabel")
    m = Model()
    x = m.variable(lower=-10.0, upper=10.0)
    m.add(abs(x - 2.0) <= 0.25)
    m.minimize(huber(x))
    compiled = m.compile()
    result = compiled.solve()
    assert compiled.reconstruct_primal(result.x)[0] == pytest.approx(1.75, abs=2e-6)
    assert compiled.validate_original(m, result.x, atol=2e-6).valid
    bad = result.x.copy()
    bad[0] = 0.0
    assert not compiled.validate_original(m, bad).valid


def test_clarabel_callback_error_and_next_solve_are_clean():
    pytest.importorskip("clarabel")
    from solverpilot.conic import ClarabelBackend

    m = Model()
    x = m.variable(2, lower=1.0, upper=1.0)
    m.minimize(norm(x))
    compiled = m.compile()
    backend = ClarabelBackend(reuse=True)

    def broken(event):
        raise ValueError("application callback failure")

    with pytest.raises(RuntimeError, match="callback"):
        compiled.solve(backend=backend, progress=broken)
    result = compiled.solve(backend=backend)
    assert result.validated
    seen = []

    def stop(event):
        seen.append(event)
        return True

    result = compiled.solve(backend=backend, progress=stop)
    assert seen and result.backend_status == "CallbackTerminated"

    def reenter(event):
        compiled.solve(backend=backend)

    with pytest.raises(RuntimeError, match="callback"):
        compiled.solve(backend=backend, progress=reenter)
    assert compiled.solve(backend=backend).validated


def test_scenario_cancelled_generator_is_not_advanced():
    m = production_model([1.0], [[1.0]], [2.0])
    before = m.data_hash
    token = CancellationToken()
    token.cancel()

    def source():
        raise AssertionError("advanced after cancellation")
        yield {}

    assert list(scenario_sweep(m, source(), cancellation=token)) == []
    assert m.data_hash == before


def test_affine_broadcast_transpose_sum_and_zero_pattern_refresh():
    m = Model()
    x = m.variable((2, 3), lower=0, upper=3)
    weights = m.parameter(3, value=[1, 0, 2], name="weights")
    m.add((x * weights).T.sum(axis=1) <= 5)
    m.minimize(x.sum())
    initial = m.compile()
    weights.value = [0, 2, 1]
    updated = m.compile()
    values = np.arange(6).reshape(2, 3)
    assert updated.execution_ir.A @ values.reshape(-1) == pytest.approx(
        (values * [0, 2, 1]).T.sum(axis=1)
    )
    weights.value = [1, 0, 2]
    assert m.compile().execution_ir.data_hash == initial.execution_ir.data_hash
