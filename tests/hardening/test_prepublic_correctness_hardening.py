from __future__ import annotations

import numpy as np
import pytest
from scipy import sparse

import solverpilot as sp
from solverpilot.backends import ScipyHighsBackend
from solverpilot.benchmark.summary import summarize_rows
from solverpilot.bridges.validation import validate_semantic_primal
from solverpilot.conic import ConeAffineBlock, ConeKind, ConicProblem, validate_conic_solution
from solverpilot.cp.ir import CPIntVarIR, CPProblem
from solverpilot.cp.reference import ReferenceCPBackend
from solverpilot.nlp.compiler import compile_nlp_model
from solverpilot.nlp.validation import validate_nlp_solution
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.validate import ValidationTolerances


def _qp(P: np.ndarray) -> QuadraticProblem:
    n = P.shape[0]
    return QuadraticProblem.from_data(
        P=P,
        A=np.zeros((0, n)),
        q=np.zeros(n),
        variable_lower=-np.ones(n),
        variable_upper=np.ones(n),
        constraint_lower=np.zeros(0),
        constraint_upper=np.zeros(0),
    )


def test_qp_large_unrelated_scale_cannot_hide_negative_curvature() -> None:
    with pytest.raises(ValueError, match="positive semidefinite"):
        _qp(np.diag([1e12, -999.0]))


def test_qp_large_unrelated_scale_cannot_hide_asymmetric_pair() -> None:
    P = np.array(
        [
            [1e12, 0.0, 0.0],
            [0.0, 2.0, 1.0],
            [0.0, 0.0, 2.0],
        ]
    )
    with pytest.raises(ValueError, match="symmetric"):
        _qp(P)


def test_qp_accepted_tiny_asymmetry_is_canonicalized_once() -> None:
    P = np.array([[2.0, 1.0 + 1e-13], [1.0, 2.0]])
    problem = _qp(P)
    dense = problem.P.toarray()
    assert np.array_equal(dense, dense.T)
    assert dense[0, 1] == pytest.approx(1.0 + 0.5e-13)


def _dummy_soc(n: int) -> ConeAffineBlock:
    return ConeAffineBlock(
        ConeKind.SECOND_ORDER,
        sparse.csr_matrix((2, n), dtype=float),
        np.array([1.0, 0.0]),
        (2,),
    )


def test_conic_unrelated_huge_variable_cannot_relax_other_variable_bound() -> None:
    problem = ConicProblem.from_data(
        P=np.zeros((2, 2)),
        q=np.zeros(2),
        A=np.zeros((0, 2)),
        variable_lower=np.array([-np.inf, 0.0]),
        variable_upper=np.array([np.inf, 0.0]),
        constraint_lower=np.zeros(0),
        constraint_upper=np.zeros(0),
        cones=[_dummy_soc(2)],
    )
    report = validate_conic_solution(problem, np.array([1e12, 1000.0]))
    assert report.max_variable_violation == pytest.approx(1000.0)
    assert not report.valid


def test_conic_psd_unrelated_huge_eigenvalue_cannot_hide_negative_eigenvalue() -> None:
    # F x -> [[x0, 0], [0, x1]]
    F = sparse.csr_matrix([[1, 0], [0, 0], [0, 0], [0, 1]], dtype=float)
    block = ConeAffineBlock(ConeKind.POSITIVE_SEMIDEFINITE, F, np.zeros(4), (2, 2))
    problem = ConicProblem.from_data(
        P=np.zeros((2, 2)),
        q=np.zeros(2),
        A=np.zeros((0, 2)),
        variable_lower=np.array([-np.inf, -np.inf]),
        variable_upper=np.array([np.inf, np.inf]),
        constraint_lower=np.zeros(0),
        constraint_upper=np.zeros(0),
        cones=[block],
    )
    report = validate_conic_solution(problem, np.array([1e12, -10000.0]))
    assert report.max_cone_violation >= 10000.0
    assert not report.valid
    assert not report.cone_checks[0].valid


def test_semantic_primal_rejects_nan_before_constraint_arithmetic() -> None:
    model = sp.Model()
    x = model.variable(lower=0.0, upper=1.0)
    model.minimize(x)
    report = validate_semantic_primal(model, [np.nan])
    assert not report.valid
    assert report.max_violation == float("inf")


def test_nlp_nonfinite_reported_objective_is_never_consistent() -> None:
    model = sp.Model()
    x = model.variable(lower=-1.0, upper=1.0)
    model.minimize(x * x)
    problem = compile_nlp_model(model).execution_ir
    for reported in (np.inf, -np.inf, np.nan):
        report = validate_nlp_solution(problem, np.array([0.0]), objective_reported=reported)
        assert not report.valid
        assert report.objective_consistent is False
        assert report.objective_difference == float("inf")


def test_linear_objective_and_bound_infinities_fail_closed() -> None:
    base = dict(
        A=np.zeros((0, 1)),
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[],
        constraint_upper=[],
    )
    for c in ([np.inf], [-np.inf]):
        with pytest.raises(ValueError, match="finite"):
            LinearProblem.from_data(c=c, **base)
    with pytest.raises(ValueError, match=r"lower bound.*\+infinity"):
        LinearProblem.from_data(c=[0.0], **{**base, "variable_lower": [np.inf]})
    with pytest.raises(ValueError, match=r"upper bound.*-infinity"):
        LinearProblem.from_data(c=[0.0], **{**base, "variable_upper": [-np.inf]})


def test_validation_tolerances_reject_nan_inf_and_non_numeric_values() -> None:
    for value in (np.nan, np.inf, -np.inf, True, "1e-7"):
        with pytest.raises(ValueError):
            ValidationTolerances(feasibility=value)  # type: ignore[arg-type]


def test_solve_result_is_a_deep_immutable_snapshot() -> None:
    problem = LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[1.0],
        constraint_lower=[0.0],
        constraint_upper=[np.inf],
    )
    result = sp.execute(problem, ScipyHighsBackend())
    assert result.x is not None
    with pytest.raises(ValueError):
        result.x[0] = 999.0
    trust = result.raw_statistics["solverpilot_trust"]  # type: ignore[index]
    with pytest.raises(TypeError):
        trust["dual_verified"] = True  # type: ignore[index]
    assert not result.optimality_evidence.independently_verified_optimal


def test_cp_result_assignment_and_statistics_are_immutable_snapshots() -> None:
    problem = CPProblem((CPIntVarIR(1, "x", (0, 1), "bool"),), (), ())
    result = ReferenceCPBackend().solve(problem)
    assert result.assignment is not None
    with pytest.raises(TypeError):
        result.assignment[1] = 99  # type: ignore[index]
    with pytest.raises(TypeError):
        result.raw_statistics["states_examined"] = 999  # type: ignore[index]


def _bench_row(run_id: str, backend: str, sha: str, *, proof: bool, wall: float = 0.1) -> dict:
    return {
        "run_id": run_id,
        "protocol_id": "p",
        "environment_id": "e",
        "instance_sha256": sha,
        "instance": "same.mps",
        "backend": backend,
        "repetition": 0,
        "state": "solved",
        "validated": True,
        "reference_check": "not_checkable",
        "public_status": "valid_optimal",
        "optimality_evidence": {"independently_verified_optimal": proof},
        "wall_s": wall,
        "trace": {},
    }


def test_benchmark_summary_rejects_same_name_with_different_instance_bytes() -> None:
    rows = [
        _bench_row("a", "a", "a" * 64, proof=True),
        _bench_row("b", "b", "b" * 64, proof=True),
    ]
    with pytest.raises(ValueError, match="instance identity mismatch"):
        summarize_rows(rows, cutoff_s=1.0)


def test_reference_free_valid_optimal_needs_independent_evidence_for_exact_metrics() -> None:
    no_proof = _bench_row("a", "unproved", "a" * 64, proof=False, wall=0.01)
    proof = _bench_row("b", "proved", "a" * 64, proof=True, wall=0.02)
    out = summarize_rows([no_proof, proof], cutoff_s=1.0, par_penalty=10.0, bootstrap_draws=10)
    assert out["solver_stats"]["unproved"]["reference_ok_rows"] == 0
    assert out["solver_stats"]["proved"]["reference_ok_rows"] == 1
    assert out["sbs_solver"] == "proved"


def test_nonfinite_benchmark_runtime_cannot_be_clipped_into_success() -> None:
    bad = _bench_row("a", "bad", "a" * 64, proof=True, wall=float("inf"))
    good = _bench_row("b", "good", "a" * 64, proof=True, wall=0.2)
    out = summarize_rows([bad, good], cutoff_s=1.0, par_penalty=10.0, bootstrap_draws=10)
    assert out["solver_stats"]["bad"]["reference_ok_rows"] == 1
    assert out["solver_stats"]["bad"]["par_cost_mean"] == pytest.approx(10.0)
    assert out["sbs_solver"] == "good"
