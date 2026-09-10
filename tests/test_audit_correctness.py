"""Regression cases from the September 2026 independent source audit."""
import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem, QuadraticProblem, Session
from solverpilot.problem import ConvexityStatus
from solverpilot.validate import CandidateSolution, validate_solution
from solverpilot.conic import ConeAffineBlock, ConeKind, ConicProblem, validate_conic_solution


def linear(**changes):
    data = dict(A=np.zeros((0, 1)), c=[0.0], variable_lower=[-np.inf],
                variable_upper=[np.inf], constraint_lower=[], constraint_upper=[])
    data.update(changes)
    return LinearProblem.from_data(**data)


@pytest.mark.parametrize("lower,upper,x", [(0., 1e12, -1.), (-1e12, 0., 1.)])
@pytest.mark.parametrize("row", [False, True])
def test_opposite_bound_cannot_relax_violated_bound(lower, upper, x, row):
    p = linear(**(dict(A=[[1.]], constraint_lower=[lower], constraint_upper=[upper])
                  if row else dict(variable_lower=[lower], variable_upper=[upper])))
    assert not validate_solution(p, CandidateSolution(np.array([x]))).valid


@pytest.mark.parametrize("reported", [None, 0.])
def test_objective_overflow_is_invalid(reported):
    with np.errstate(over="ignore", invalid="ignore"):
        report = validate_solution(linear(c=[1e308]), CandidateSolution(np.array([2.]), reported))
    assert not report.valid


def test_activity_overflow_is_invalid():
    p = linear(A=[[1e308]], constraint_lower=[0.], constraint_upper=[1.])
    with np.errstate(over="ignore", invalid="ignore"):
        assert not validate_solution(p, CandidateSolution(np.array([2.]))).valid


def test_sparse_duplicate_sum_overflow_rejected():
    a = sparse.csr_matrix(([1e308, 1e308], [0, 0], [0, 2]), shape=(1, 1))
    with np.errstate(over="ignore"), pytest.raises(ValueError, match="finite"):
        linear(A=a, constraint_lower=[0.], constraint_upper=[1.])


def test_large_finite_hessian_does_not_overflow_during_symmetrization():
    with np.errstate(over="ignore"):
        p = QuadraticProblem(linear(), sparse.csr_matrix([[1e308]]), ConvexityStatus.CONFIRMED)
    assert p.P[0, 0] == 1e308


def test_symmetric_subnormal_hessian_is_preserved():
    smallest = np.nextafter(0., 1.)
    p = QuadraticProblem(linear(), sparse.csr_matrix([[smallest]]))
    assert p.P[0, 0] == smallest


def test_direct_qp_maximization_rejected():
    with pytest.raises(ValueError, match="minimization"):
        QuadraticProblem(linear(objective_sense="maximize"), sparse.eye(1), ConvexityStatus.CONFIRMED)


def test_session_qp_maximization_update_is_atomic():
    session = Session(QuadraticProblem(linear(), sparse.eye(1), ConvexityStatus.CONFIRMED))
    original = session.problem
    with pytest.raises(ValueError, match="minimization"):
        session.update(objective_sense="maximize")
    assert session.problem is original and session.revision == 0


def conic(value, kind=ConeKind.SECOND_ORDER, **changes):
    value = np.asarray(value, dtype=float)
    block = ConeAffineBlock(kind, sparse.csr_matrix((value.size, 1)), value.ravel(), value.shape)
    data = dict(P=[[0.]], q=[0.], A=np.zeros((0, 1)), variable_lower=[-np.inf],
                variable_upper=[np.inf], constraint_lower=[], constraint_upper=[], cones=[block])
    data.update(changes)
    return ConicProblem.from_data(**data)


def test_conic_opposite_bound_cannot_mask_violation():
    p = conic([1., 0.], variable_lower=[0.], variable_upper=[1e12])
    assert not validate_conic_solution(p, np.array([-1.])).valid


def test_soc_overflow_cannot_validate_infeasible_point():
    with np.errstate(over="ignore", invalid="ignore"):
        assert not validate_conic_solution(conic([0., 1e308]), np.array([0.])).valid


def test_rsoc_negative_coordinate_not_masked_by_squared_scale():
    p = conic([-1e12, -1e12, 0.], ConeKind.ROTATED_SECOND_ORDER)
    assert not validate_conic_solution(p, np.array([0.])).valid


def test_rsoc_overflow_cannot_validate_infeasible_point():
    with np.errstate(over="ignore", invalid="ignore"):
        p = conic([1e200, 1e200, 2e200], ConeKind.ROTATED_SECOND_ORDER)
        assert not validate_conic_solution(p, np.array([0.])).valid


def test_conic_objective_overflow_invalid():
    with np.errstate(over="ignore", invalid="ignore"):
        assert not validate_conic_solution(conic([1., 0.], q=[1e308]), np.array([2.])).valid


def test_session_metadata_update_not_silently_discarded():
    session = Session(linear())
    original_hash = session.problem.data_hash
    session.update(name="renamed", metadata={"origin": "audit"})
    assert session.problem.name == "renamed"
    assert session.problem.metadata == {"origin": "audit"}
    assert session.problem.data_hash == original_hash
    assert session.revision == 0


@pytest.mark.parametrize("claimed", [True, "false"])
def test_backend_statistics_cannot_self_certify_optimality(claimed):
    from solverpilot import execute
    from solverpilot.backends import ScipyHighsBackend, BackendSolveResult

    class ClaimingBackend(ScipyHighsBackend):
        def solve(self, problem):
            return BackendSolveResult("optimal", np.array([0.]), 0., {
                "dual_optimality_verified": claimed,
                "optimality_gap_verified": claimed,
                "optimality_certificate_verified": claimed,
            })

    result = execute(linear(), ClaimingBackend())
    assert result.optimality_evidence.primal_validated
    assert not result.optimality_evidence.independently_verified_optimal


@pytest.mark.parametrize("value,kind", [
    ([1e200, 1e200], ConeKind.SECOND_ORDER),
    ([1e200, 1e200, 1e200], ConeKind.ROTATED_SECOND_ORDER),
])
def test_large_valid_cones_remain_valid(value, kind):
    assert validate_conic_solution(conic(value, kind), np.array([0.])).valid
