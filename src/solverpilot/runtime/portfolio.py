from __future__ import annotations

from dataclasses import dataclass
from time import perf_counter
from typing import Iterable

from solverpilot.backends import Backend, BackendHealthReport, BackendProbeStatus
from solverpilot.plan import SolveBudget
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem
from solverpilot.validate import PublicStatus

from .budgeting import apply_budget
from .executor import execute
from .result import SolveResult


_VALID_SOLUTION_STATUSES = {
    PublicStatus.VALID_OPTIMAL,
    PublicStatus.VALID_FEASIBLE,
    PublicStatus.FEASIBLE_LIMIT,
}


@dataclass(frozen=True, slots=True)
class PortfolioAttempt:
    backend: str
    status: str | None
    objective: float | None
    validation_valid: bool | None
    duration_s: float
    error: str | None = None
    skipped_reason: str | None = None


@dataclass(frozen=True, slots=True)
class PortfolioSolveResult:
    chosen: SolveResult | None
    attempts: tuple[PortfolioAttempt, ...]
    total_s: float
    stopped_reason: str
    budget_exhausted: bool

    @property
    def chosen_backend(self) -> str | None:
        return None if self.chosen is None else self.chosen.trace.backend


def _better(
    problem: LinearProblem | QuadraticProblem,
    incumbent: SolveResult | None,
    candidate: SolveResult,
) -> bool:
    if candidate.status not in _VALID_SOLUTION_STATUSES or candidate.objective is None:
        return False
    if incumbent is None or incumbent.status not in _VALID_SOLUTION_STATUSES or incumbent.objective is None:
        return True
    if problem.objective_sense is ObjectiveSense.MINIMIZE:
        return float(candidate.objective) < float(incumbent.objective)
    return float(candidate.objective) > float(incumbent.objective)


def _health_map(reports: Iterable[BackendHealthReport] | None) -> dict[str, BackendHealthReport]:
    return {} if reports is None else {r.backend: r for r in reports}


def execute_portfolio(
    problem: LinearProblem | QuadraticProblem,
    backends: Iterable[Backend],
    *,
    budget: SolveBudget | None = None,
    stop_on_valid_optimal: bool = True,
    health_reports: Iterable[BackendHealthReport] | None = None,
    require_healthy: bool = False,
) -> PortfolioSolveResult:
    """Execute a sequential validated backend portfolio.

    The portfolio is deliberately conservative: every candidate goes through the same
    capability checks and independent validator as :func:`execute`; exceptions are
    isolated to one attempt. If a global wall-time budget is supplied, remaining time
    is passed to each backend and backends that cannot enforce it are skipped rather
    than silently violating the budget.

    This is an execution/fallback primitive, not an empirical solver-selection policy.
    """

    t0 = perf_counter()
    attempts: list[PortfolioAttempt] = []
    chosen: SolveResult | None = None
    terminal_without_solution: list[SolveResult] = []
    health = _health_map(health_reports)
    stopped_reason = "portfolio_exhausted"
    budget_exhausted = False

    for backend in tuple(backends):
        name = backend.manifest.name
        report = health.get(name)
        if report is not None and report.version != backend.manifest.version:
            report = None
        if report is not None and report.status in {BackendProbeStatus.UNHEALTHY, BackendProbeStatus.UNAVAILABLE}:
            attempts.append(
                PortfolioAttempt(
                    backend=name,
                    status=None,
                    objective=None,
                    validation_valid=None,
                    duration_s=0.0,
                    skipped_reason=f"health:{report.status.value}",
                )
            )
            continue
        if require_healthy and (report is None or report.status is not BackendProbeStatus.HEALTHY):
            attempts.append(
                PortfolioAttempt(
                    backend=name,
                    status=None,
                    objective=None,
                    validation_valid=None,
                    duration_s=0.0,
                    skipped_reason="health:not_confirmed_healthy",
                )
            )
            continue

        elapsed = perf_counter() - t0
        remaining: float | None = None
        if budget is not None and budget.wall_time_s is not None:
            remaining = float(budget.wall_time_s) - elapsed
            if remaining <= 0:
                budget_exhausted = True
                stopped_reason = "budget_exhausted"
                break

        attempt_budget = budget
        if budget is not None and remaining is not None:
            attempt_budget = SolveBudget(
                wall_time_s=remaining,
                memory_mb=budget.memory_mb,
                threads=budget.threads,
            )

        a0 = perf_counter()
        try:
            configured = apply_budget(backend, attempt_budget)
            result = execute(problem, configured)
            duration = perf_counter() - a0
            attempts.append(
                PortfolioAttempt(
                    backend=name,
                    status=result.status.value,
                    objective=None if result.objective is None else float(result.objective),
                    validation_valid=None if result.validation is None else bool(result.validation.valid),
                    duration_s=duration,
                )
            )

            if _better(problem, chosen, result):
                chosen = result
            elif chosen is None and result.status in {
                PublicStatus.INFEASIBLE,
                PublicStatus.UNBOUNDED,
                PublicStatus.INFEASIBLE_OR_UNBOUNDED,
            }:
                terminal_without_solution.append(result)

            if stop_on_valid_optimal and result.status is PublicStatus.VALID_OPTIMAL:
                chosen = result
                stopped_reason = "valid_optimal_found"
                break
        except Exception as exc:
            attempts.append(
                PortfolioAttempt(
                    backend=name,
                    status=None,
                    objective=None,
                    validation_valid=None,
                    duration_s=perf_counter() - a0,
                    error=f"{type(exc).__name__}: {exc}",
                )
            )
            continue

    if chosen is None and terminal_without_solution:
        terminal_statuses = {r.status for r in terminal_without_solution}
        if len(terminal_statuses) == 1:
            chosen = terminal_without_solution[0]
        else:
            stopped_reason = "conflicting_terminal_statuses"
    total_s = perf_counter() - t0
    if not attempts:
        stopped_reason = "no_attempts"
    return PortfolioSolveResult(
        chosen=chosen,
        attempts=tuple(attempts),
        total_s=total_s,
        stopped_reason=stopped_reason,
        budget_exhausted=budget_exhausted,
    )
