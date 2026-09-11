"""Static contract for the additive, typed common result and options API."""

from typing import Any, assert_type
from solverpilot.runtime.options import SolveOptions, ProgressEvent
from solverpilot.runtime.unified import SolutionSummary, summarize, solve_any, solve_verified
from solverpilot.plan import SolveBudget


def consume(problem: Any, result: Any) -> None:
    options = SolveOptions(budget=SolveBudget(wall_time_s=10.0))
    assert_type(options, SolveOptions)
    assert_type(summarize(result), SolutionSummary)
    assert_type(solve_any(problem, options=options), tuple[SolutionSummary, Any])
    assert_type(solve_verified(problem, options=options), tuple[SolutionSummary, Any])
    event = ProgressEvent("clarabel-native", "iteration", iteration=1)
    assert_type(event.iteration, int | None)
