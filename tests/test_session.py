import numpy as np
import pytest

from solverpilot import LinearProblem, VariableDomain
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.session import MutationKind, Session, assess_reuse
from solverpilot.validate import PublicStatus


def base_lp():
    return LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )


def test_session_objective_update_tracks_revision_and_solves():
    s = Session(base_lp())
    r0 = s.solve()
    assert r0.status is PublicStatus.VALID_OPTIMAL
    assert s.revision == 0

    change = s.update(c=[3.0, 1.0])
    assert change.kinds == (MutationKind.OBJECTIVE_VECTOR,)
    assert not change.structural_change
    assert s.revision == 1

    r1 = s.solve()
    assert r1.status is PublicStatus.VALID_OPTIMAL
    assert r1.session_revision == 1
    assert r1.objective == pytest.approx(1.0)
    # Current development bridges do not pretend to reuse solver state.
    assert s.last_reuse_assessment is not None
    assert not s.last_reuse_assessment.applied
    assert not s.last_reuse_assessment.safe_candidate


def test_session_noop_update_does_not_increment_revision():
    s = Session(base_lp())
    change = s.update(c=[1.0, 2.0])
    assert change.is_noop
    assert s.revision == 0
    assert s.history == ()


def test_same_sparsity_matrix_value_update_classification():
    s = Session(base_lp())
    change = s.update(A=[[2.0, 1.0]])
    assert MutationKind.MATRIX_VALUES_SAME_SPARSITY in change.kinds
    assert MutationKind.MATRIX_SPARSITY_CHANGED not in change.kinds


def test_sparsity_change_classification():
    s = Session(base_lp())
    change = s.update(A=[[1.0, 0.0]])
    assert MutationKind.MATRIX_SPARSITY_CHANGED in change.kinds
    assert change.structural_change


def test_integrality_change_classification():
    s = Session(base_lp())
    change = s.update(domains=[VariableDomain.BINARY, VariableDomain.CONTINUOUS])
    assert MutationKind.INTEGRALITY_CHANGED in change.kinds
    assert change.structural_change


def test_reuse_assessment_distinguishes_possible_from_applied():
    s = Session(base_lp())
    record = s.update(c=[2.0, 1.0])
    manifest = BackendManifest(
        name="fake-native",
        capabilities={
            Capability.LP: SupportLevel.NATIVE,
            Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE,
            Capability.BASIS_START: SupportLevel.NATIVE,
        },
    )
    assessment = assess_reuse(s.problem, manifest, (record,))
    assert assessment.safe_candidate
    assert Capability.SAME_SPARSITY_DATA_UPDATE in assessment.supported
    assert Capability.BASIS_START in assessment.supported
    assert assessment.applied is False


def test_update_unknown_field_is_error():
    s = Session(base_lp())
    with pytest.raises(TypeError, match="unknown"):
        s.update(does_not_exist=1)


def test_session_records_backend_reported_reuse_as_applied():
    from solverpilot.backends import BackendRegistry, BackendSolveResult

    class ReusingBackend:
        def __init__(self):
            self.calls = 0

        @property
        def manifest(self):
            return BackendManifest(
                name="fake-reusing",
                capabilities={
                    Capability.LP: SupportLevel.NATIVE,
                    Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE,
                },
            )

        def is_available(self):
            return True

        def solve(self, problem):
            self.calls += 1
            # The LP min x0 + 2*x1, x0+x1>=1 has optimum [1,0]. After the
            # test objective update to [2,1], optimum is [0,1].
            x = np.array([1.0, 0.0]) if self.calls == 1 else np.array([0.0, 1.0])
            return BackendSolveResult(
                backend_status="optimal",
                x=x,
                objective_reported=float(problem.c @ x + problem.objective_offset),
                raw_statistics={
                    "reuse_applied": self.calls > 1,
                    "reuse_mode": "fake_same_sparsity_update",
                },
            )

    backend = ReusingBackend()
    reg = BackendRegistry()
    reg.register(backend)
    s = Session(base_lp(), registry=reg)
    assert s.solve(backend=backend).status is PublicStatus.VALID_OPTIMAL
    s.update(c=[2.0, 1.0])
    assert s.solve(backend=backend).status is PublicStatus.VALID_OPTIMAL
    assert s.last_reuse_assessment is not None
    assert s.last_reuse_assessment.applied
    assert "fake_same_sparsity_update" in s.last_reuse_assessment.reason
