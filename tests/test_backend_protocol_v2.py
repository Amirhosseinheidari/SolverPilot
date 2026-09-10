import numpy as np

from solverpilot.backends import (
    BackendArtifactBundle,
    BackendSolveRequestV2,
    BackendSolveResultV2,
    CallbackEvent,
    CallbackEventKind,
)


def test_backend_artifacts_are_independent_optional_channels():
    art = BackendArtifactBundle(primal=np.array([1.0]), objective=2.0)
    assert art.primal.tolist() == [1.0]
    assert art.dual is None
    assert art.basis is None
    assert art.solution_pool == ()


def test_backend_solve_request_keeps_start_types_separate():
    req = BackendSolveRequestV2(
        primal_start=np.array([1.0, 2.0]),
        dual_start=np.array([3.0]),
        basis_start={"columns": ["basic"]},
        mip_start=np.array([1.0, 0.0]),
        time_limit_s=10.0,
        thread_limit=1,
    )
    assert req.primal_start.shape == (2,)
    assert req.dual_start.shape == (1,)
    assert req.basis_start is not None
    assert req.mip_start.shape == (2,)


def test_callback_event_does_not_imply_control_surface():
    event = CallbackEvent(CallbackEventKind.PROGRESS, runtime_s=1.0, objective=4.0, bound=3.0)
    assert event.kind is CallbackEventKind.PROGRESS
    assert event.objective == 4.0
    assert event.bound == 3.0


def test_backend_result_v2_separates_termination_from_artifacts():
    result = BackendSolveResultV2(
        termination="solver_reported_optimal",
        artifacts=BackendArtifactBundle(primal=None, objective=None),
    )
    assert result.termination == "solver_reported_optimal"
    assert result.artifacts.primal is None
