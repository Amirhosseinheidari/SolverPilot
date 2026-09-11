import numpy as np
import pytest
from solverpilot import LinearProblem
from solverpilot.validate import verify_infeasibility, verify_unboundedness, verify_optimality


@pytest.mark.parametrize("scale", [1e-14, 1e-10, 1e-4, 1.0, 1e4, 1e10])
def test_certificates_preserve_meaning_under_row_scaling(scale):
    feasible = LinearProblem.from_data(
        A=[[scale]],
        c=[0.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
    )
    assert not verify_infeasibility(feasible, [-1.0, 0.0])
    bounded = LinearProblem.from_data(
        A=[[scale]],
        c=[-1.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[-np.inf],
        constraint_upper=[1.0],
    )
    assert not verify_unboundedness(bounded, [0.0], [1.0])
    impossible = LinearProblem.from_data(
        A=[[scale]],
        c=[0.0],
        variable_lower=[0.0],
        variable_upper=[0.0],
        constraint_lower=[scale],
        constraint_upper=[np.inf],
    )
    assert verify_infeasibility(impossible, [-1.0, scale], atol=0.0)


@pytest.mark.parametrize("scale", [1e-12, 1e-8, 1e-4, 1.0, 1e4, 1e8])
def test_optimality_gap_accounts_for_variable_units(scale):
    p = LinearProblem.from_data(
        A=np.empty((0, 2)),
        c=[scale, -scale],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0 / scale, 1.0 / scale],
        constraint_lower=[],
        constraint_upper=[],
    )
    checked = verify_optimality(p, [0.0, 0.0], [0.0, 0.0])
    assert not checked.verified and checked.gap == pytest.approx(1.0)
    assert verify_optimality(p, [0.0, 1.0 / scale], [-scale, scale]).verified


def test_random_box_oracle_survives_variable_permutations():
    rng = np.random.default_rng(730)
    for _ in range(25):
        cost = rng.integers(-10, 10, 8).astype(float)
        lower, upper = -rng.uniform(1, 3, 8), rng.uniform(1, 3, 8)
        order = rng.permutation(8)
        p = LinearProblem.from_data(
            A=np.empty((0, 8)),
            c=cost[order],
            variable_lower=lower[order],
            variable_upper=upper[order],
            constraint_lower=[],
            constraint_upper=[],
        )
        best = np.where(cost > 0, lower, upper)[order]
        assert verify_optimality(p, best, -cost[order]).verified


def test_non_numeric_certificate_is_unverified():
    p = LinearProblem.from_data(
        A=np.empty((0, 1)),
        c=[-1.0],
        variable_lower=[0.0],
        variable_upper=[np.inf],
        constraint_lower=[],
        constraint_upper=[],
    )
    assert not verify_infeasibility(p, ["invalid"])
    assert not verify_unboundedness(p, [0.0], ["invalid"])


def test_stationarity_cancellation_does_not_erase_unbounded_box_error():
    # In ordinary binary64 arithmetic (1e16 + 1) - 1e16 loses the unit.
    p = LinearProblem.from_data(
        A=[[1e16], [1.0], [-1e16]],
        c=[0.0],
        variable_lower=[-np.inf],
        variable_upper=[np.inf],
        constraint_lower=[-np.inf] * 3,
        constraint_upper=[0.0, -1.0, 0.0],
    )
    assert not verify_infeasibility(p, [1.0, 1.0, 1.0, 0.0])
