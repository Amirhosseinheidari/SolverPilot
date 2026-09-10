from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solverpilot import LinearProblem, ObjectiveSense, SolveBudget, VariableDomain, execute_portfolio
from solverpilot.backends import BackendSolveResult, ScipyHighsLPBackend, probe_backends
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel


def min_lp():
    return LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[1.0], constraint_upper=[2.0],
    )


def max_lp():
    return LinearProblem.from_data(
        A=[[1.0]], c=[1.0],
        variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[0.0], constraint_upper=[2.0],
        objective_sense=ObjectiveSense.MAXIMIZE,
    )


@dataclass(slots=True)
class FakeLPBackend:
    name: str
    x: float | None = None
    status: str = "optimal"
    raises: bool = False
    time_limit_s: float | None = None

    @property
    def manifest(self):
        return BackendManifest(
            name=self.name,
            version="test",
            capabilities={Capability.LP: SupportLevel.NATIVE},
        )

    def is_available(self):
        return True

    def solve(self, problem):
        if self.raises:
            raise RuntimeError("intentional backend failure")
        x = None if self.x is None else np.asarray([self.x], dtype=float)
        objective = None if x is None else float(problem.c @ x + problem.objective_offset)
        return BackendSolveResult(self.status, x, objective, {})


def test_portfolio_isolates_failure_and_chooses_later_validated_solution():
    result = execute_portfolio(
        min_lp(),
        [
            FakeLPBackend("boom", raises=True),
            FakeLPBackend("invalid", x=0.0),
            FakeLPBackend("good", x=1.0),
        ],
    )
    assert result.chosen is not None
    assert result.chosen_backend == "good"
    assert result.chosen.status.value == "valid_optimal"
    assert result.attempts[0].error and "intentional" in result.attempts[0].error
    assert result.attempts[1].status == "invalid_solution"
    assert result.attempts[2].validation_valid is True


def test_portfolio_selects_best_validated_feasible_for_min_and_max():
    min_result = execute_portfolio(
        min_lp(),
        [
            FakeLPBackend("feasible15", x=1.5, status="converged_candidate"),
            FakeLPBackend("feasible12", x=1.2, status="converged_candidate"),
        ],
        stop_on_valid_optimal=False,
    )
    assert min_result.chosen_backend == "feasible12"
    assert min_result.chosen.objective == 1.2

    max_result = execute_portfolio(
        max_lp(),
        [
            FakeLPBackend("feasible12", x=1.2, status="converged_candidate"),
            FakeLPBackend("feasible18", x=1.8, status="converged_candidate"),
        ],
        stop_on_valid_optimal=False,
    )
    assert max_result.chosen_backend == "feasible18"
    assert max_result.chosen.objective == 1.8


def test_portfolio_can_require_active_health():
    real = ScipyHighsLPBackend("highs-ds")
    reports = probe_backends([real])
    result = execute_portfolio(
        min_lp(),
        [FakeLPBackend("unprobed", x=1.0), real],
        health_reports=reports,
        require_healthy=True,
    )
    assert result.attempts[0].skipped_reason == "health:not_confirmed_healthy"
    assert result.chosen_backend == "scipy-highs-ds"


def test_portfolio_budget_is_not_silently_ignored():
    # Fake dataclass has time_limit_s, so the budget can be cloned/enforced by contract.
    result = execute_portfolio(
        min_lp(),
        [FakeLPBackend("budgeted", x=1.0)],
        budget=SolveBudget(wall_time_s=1.0),
    )
    assert result.chosen_backend == "budgeted"
    assert not result.budget_exhausted


def test_portfolio_does_not_hide_conflicting_terminal_claims():
    result = execute_portfolio(
        min_lp(),
        [
            FakeLPBackend("claims-infeasible", x=None, status="infeasible"),
            FakeLPBackend("claims-unbounded", x=None, status="unbounded"),
        ],
        stop_on_valid_optimal=False,
    )
    assert result.chosen is None
    assert result.stopped_reason == "conflicting_terminal_statuses"


def test_portfolio_can_return_consistent_terminal_consensus():
    result = execute_portfolio(
        min_lp(),
        [
            FakeLPBackend("inf1", x=None, status="infeasible"),
            FakeLPBackend("inf2", x=None, status="infeasible"),
        ],
        stop_on_valid_optimal=False,
    )
    assert result.chosen is not None
    assert result.chosen.status.value == "infeasible"


def test_portfolio_does_not_trust_stale_health_report_version():
    real = ScipyHighsLPBackend("highs-ds")
    report = probe_backends([real])[0]
    from dataclasses import replace
    stale = replace(report, version="stale-version")
    result = execute_portfolio(
        min_lp(), [real],
        health_reports=[stale],
        require_healthy=True,
    )
    assert result.chosen is None
    assert result.attempts[0].skipped_reason == "health:not_confirmed_healthy"


def test_qp_portfolio_records_exactly_one_attempt_per_backend():
    import numpy as np
    from solverpilot import QuadraticProblem, execute_portfolio
    from solverpilot.backends import NLoptNativeBackend, ScipySLSQPQPBackend

    if not NLoptNativeBackend().is_available():
        import pytest
        pytest.skip("nlopt not installed")
    p = QuadraticProblem.from_data(
        P=[[2.0]], A=np.zeros((0, 1)), q=[-2.0],
        variable_lower=[0.0], variable_upper=[3.0],
        constraint_lower=[], constraint_upper=[],
    )
    r = execute_portfolio(
        p,
        [NLoptNativeBackend(), ScipySLSQPQPBackend()],
        stop_on_valid_optimal=False,
    )
    assert len(r.attempts) == 2
    assert all(a.error is None for a in r.attempts)
    assert r.chosen is not None
    assert r.chosen.validation is not None and r.chosen.validation.valid
