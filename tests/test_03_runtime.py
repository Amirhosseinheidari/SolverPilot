import json
import numpy as np
import pytest
from solverpilot import Model, LinearProblem, SolveBudget
from solverpilot.session import ReoptimizationSession
from solverpilot.runtime import solve
from solverpilot.runtime.batch import BatchExecutor, CancellationToken, iter_solve_batch
from solverpilot.runtime.manifest import save_run, replay_run
from solverpilot.runtime.options import SolveOptions


def problem():
    return LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )


def test_budget_preserves_workspace_and_restores_settings():
    pytest.importorskip("osqp")
    from solverpilot.backends import OSQPNativeBackend

    model = Model()
    x = model.variable(lower=-10.0, upper=10.0)
    target = model.parameter(value=1.0, name="target")
    model.minimize((x - target) ** 2)
    backend = OSQPNativeBackend()
    with ReoptimizationSession(model, backend=backend) as session:
        for i in range(3):
            result = session.solve(updates={"target": i}, budget=SolveBudget(wall_time_s=10))
            assert result.trace.reuse_applied == (i != 0)
            assert result.x[0] == pytest.approx(i, abs=1e-5)
            assert backend.time_limit_s is None
            assert result.trace.parameters["backend_configuration"]["time_limit_s"] == 10.0


def test_stream_does_not_consume_cancelled_input():
    token = CancellationToken()
    token.cancel()

    def source():
        raise AssertionError("cancelled stream consumed input")
        yield problem()

    assert list(iter_solve_batch(source(), cancellation=token)) == []


def test_stream_bounds_queue_and_reuses_process_across_calls():
    consumed = []

    def source():
        for i in range(20):
            consumed.append(i)
            yield problem()

    with BatchExecutor(
        backend="scipy-highs-ds", max_workers=1, max_pending=2, timeout_s=30
    ) as executor:
        stream = executor.iter(source())
        assert next(stream).validation_valid
        assert len(consumed) == 2
        process = executor._workers[0].process
        stream.close()
        result = list(executor.iter([problem()]))
        assert result[0].validation_valid
        # A cancelled in-flight job may replace its worker. Completed batches
        # must keep their next worker alive for reuse.
        process = executor._workers[0].process
        assert list(executor.iter([problem()]))[0].validation_valid
        assert executor._workers[0].process is process


def test_manifest_replay_and_tamper_detection(tmp_path):
    p = problem()
    result = solve(p, backend="scipy-highs-ds", budget=SolveBudget(wall_time_s=10))
    path = tmp_path / "run.json"
    save_run(path, p, result, include_model=True)
    assert replay_run(path).objective == pytest.approx(result.objective)
    payload = json.loads(path.read_text())
    payload["model"]["c"][0] = 2.0
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="hash"):
        replay_run(path)
    save_run(path, p, result)
    assert "model" not in json.loads(path.read_text())


def test_clarabel_reuse_progress_and_cancellation():
    pytest.importorskip("clarabel")
    from solverpilot.conic import ClarabelBackend

    m = Model()
    x = m.variable(2, lower=[3, 4], upper=[3, 4])
    t = m.variable(lower=0, upper=10)
    m.soc(t, x)
    m.minimize(t)
    compiled = m.compile()
    backend = ClarabelBackend(reuse=True)
    events = []
    first = compiled.solve(
        backend=backend,
        options=SolveOptions(budget=SolveBudget(wall_time_s=10), progress=events.append),
    )
    second = compiled.solve(backend=backend, budget=SolveBudget(wall_time_s=10))
    assert first.validated and second.validated and events
    assert second.raw_statistics["reuse_applied"]
    token = CancellationToken()
    token.cancel()
    stopped = compiled.solve(backend=backend, cancellation=token)
    assert stopped.backend_status == "Cancelled" and stopped.x is None
    with pytest.raises(TypeError):
        first.raw_statistics["transport_dual"][0] = 123
