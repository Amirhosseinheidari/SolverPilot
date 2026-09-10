import math

import pytest

from solverpilot.evaluation import performance_profile, portfolio_metrics


def test_portfolio_metrics_sbs_vbs_and_gap_closure():
    costs = {
        "a": [1.0, 4.0, 1.0, 4.0],
        "b": [4.0, 1.0, 4.0, 1.0],
    }
    policy = ["a", "b", "a", "b"]
    m = portfolio_metrics(costs, policy)
    assert m.sbs_solver == "a"  # equal means -> lexical tie-break
    assert m.sbs_cost == 2.5
    assert m.vbs_cost == 1.0
    assert m.policy_cost == 1.0
    assert m.gap_closure == pytest.approx(1.0)


def test_gap_closure_zero_for_sbs_policy():
    costs = {"a": [1.0, 4.0], "b": [4.0, 1.0]}
    m = portfolio_metrics(costs, ["a", "a"])
    assert m.gap_closure == pytest.approx(0.0)


def test_gap_closure_none_without_selector_opportunity():
    costs = {"a": [1.0, 1.0], "b": [1.0, 1.0]}
    m = portfolio_metrics(costs, ["a", "b"])
    assert m.gap_closure is None


def test_performance_profile():
    costs = {"a": [1.0, 4.0], "b": [2.0, 2.0]}
    profile = performance_profile(costs, [1.0, 2.0, 4.0])
    assert profile["a"] == [0.5, 1.0, 1.0]
    assert profile["b"] == [0.5, 1.0, 1.0]


def test_bad_costs_rejected():
    with pytest.raises(ValueError):
        portfolio_metrics({}, [])
    with pytest.raises(ValueError):
        portfolio_metrics({"a": [1.0], "b": [1.0, 2.0]}, ["a"])
    with pytest.raises(ValueError):
        portfolio_metrics({"a": [math.inf]}, ["a"])


def test_policy_costs_can_include_selection_overhead():
    costs = {"a": [1.0, 4.0], "b": [4.0, 1.0]}
    m = portfolio_metrics(costs, ["a", "b"], policy_costs=[1.5, 1.5])
    assert m.sbs_cost == 2.5
    assert m.vbs_cost == 1.0
    assert m.policy_cost == 1.5
    assert m.gap_closure == pytest.approx(2.0 / 3.0)
