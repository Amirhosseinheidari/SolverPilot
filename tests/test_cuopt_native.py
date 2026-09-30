"""Run explicitly on a CUDA host: SOLVERPILOT_TEST_CUOPT=1 pytest ... ."""

import os

import numpy as np
import pytest
from scipy import sparse

from solverpilot import LinearProblem, solve
from solverpilot.backends import CuOptBackend

pytestmark = [
    pytest.mark.native,
    pytest.mark.skipif(
        os.environ.get("SOLVERPILOT_TEST_CUOPT") != "1",
        reason="explicit GPU qualification host required",
    ),
]


def checked(problem):
    result = solve(problem, backend=CuOptBackend(time_limit_s=30))
    assert result.validation is not None and result.validation.valid, result
    assert result.raw_statistics["native_solved_by"] == "PDLP"
    assert result.raw_statistics["gpu_execution_reported"]
    assert result.optimality_evidence.independently_verified_optimal, result.raw_statistics
    return result


@pytest.mark.parametrize("sense", ["minimize", "maximize"])
def test_gpu_ranged_rows_and_objective_offset(sense):
    p = LinearProblem.from_data(
        A=[[1.0, 1.0], [1.0, -1.0]],
        c=[-2.0, -1.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[10.0, 10.0],
        constraint_lower=[2.0, -2.0],
        constraint_upper=[5.0, 1.0],
        objective_sense=sense,
        objective_offset=7.0,
    )
    r = checked(p)
    reference = solve(p, backend="scipy-highs-bridge")
    assert r.objective == pytest.approx(reference.objective, abs=1e-6)


def test_gpu_free_variable_equalities():
    p = LinearProblem.from_data(
        A=[[3.0, 0.0], [0.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[-np.inf, 0.0],
        variable_upper=[np.inf, 4.0],
        constraint_lower=[1.0, 2.0],
        constraint_upper=[1.0, 3.0],
    )
    r = solve(p, backend=CuOptBackend(time_limit_s=30))
    assert r.validation.valid and r.objective == pytest.approx(13 / 3, abs=1e-6)
    # On a free domain even a tiny residual may prevent a finite independently
    # verified bound. Do not equate native convergence with a certificate.
    assert r.raw_statistics["native_solved_by"] == "PDLP"


def test_gpu_bounds_only():
    p = LinearProblem.from_data(
        A=sparse.csr_matrix((0, 3)),
        c=[1.0, -1.0, 0.0],
        variable_lower=[-2.0, 0.0, 1.0],
        variable_upper=[4.0, 3.0, 1.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    assert checked(p).objective == pytest.approx(-5.0, abs=1e-6)


def test_gpu_random_lps_match_highs():
    rng = np.random.default_rng(2648)
    for index in range(8):
        a = rng.normal(size=(15, 24))
        feasible = rng.uniform(-0.5, 0.5, 24)
        activity = a @ feasible
        p = LinearProblem.from_data(
            A=a,
            c=rng.normal(size=24),
            variable_lower=np.full(24, -1.0),
            variable_upper=np.ones(24),
            constraint_lower=activity - rng.uniform(0.1, 1.0, 15),
            constraint_upper=activity + rng.uniform(0.1, 1.0, 15),
            objective_sense="maximize" if index % 2 else "minimize",
            objective_offset=index / 3,
        )
        r = checked(p)
        cpu = solve(p, backend="scipy-highs-bridge")
        assert cpu.validation.valid
        assert r.objective == pytest.approx(cpu.objective, abs=1e-5, rel=1e-6)


@pytest.mark.parametrize("infeasible", [False, True])
def test_gpu_nonoptimal_termination_never_independently_optimal(infeasible):
    p = LinearProblem.from_data(
        A=[[1.0], [1.0]],
        c=[-1.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[1.0, -np.inf] if infeasible else [-np.inf, -np.inf],
        constraint_upper=[np.inf, 0.0] if infeasible else [np.inf, np.inf],
    )
    r = solve(p, backend=CuOptBackend(time_limit_s=20, iteration_limit=10000))
    assert r.backend_status in {
        "infeasible",
        "infeasible_or_unbounded",
        "limit_no_solution",
        "limit_feasible",
        "solver_error",
    }
    assert not r.optimality_evidence.independently_verified_optimal


def test_gpu_wall_timeout_cleans_up_worker():
    p = LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[1.0],
        constraint_lower=[0.0],
        constraint_upper=[1.0],
    )
    r = CuOptBackend(time_limit_s=0.05).solve(p)
    assert r.x is None and r.backend_status == "limit_no_solution"
