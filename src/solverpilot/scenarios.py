"""Independent parameter scenarios using an owned reusable model session."""

from dataclasses import dataclass, replace, field
from time import perf_counter
from solverpilot._immutability import deep_freeze
from solverpilot.runtime.unified import summarize
from solverpilot.session import ReoptimizationSession
from solverpilot.runtime.budgeting import CallBudget
from solverpilot.runtime.options import SolveOptions


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    index: int
    name: str
    updates: object
    summary: object = None
    result: object = None
    error: str | None = None
    problem: object = None
    aggregate_budget_report: object = field(default=None,kw_only=True)

    def __post_init__(self):
        object.__setattr__(self, "updates", deep_freeze(self.updates))
        object.__setattr__(self, "aggregate_budget_report", deep_freeze(self.aggregate_budget_report))


def scenario_sweep(
    model, scenarios, *, backend=None, options=None, cancellation=None, on_error="raise", aggregate_budget=None
):
    """Yield scenarios; unspecified parameters reset to the original baseline.

    Each input is an update mapping or (name, mapping). Input is consumed lazily.
    Close the iterator if stopping early, to release its private session.
    aggregate_budget covers cumulative execution including session setup; consumer
    idle time between yielded results is excluded. options.budget stays per scenario.
    """
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be raise or record")
    tracker=CallBudget(aggregate_budget)
    baseline = {p.name: p.value for p in model.parameters}
    with ReoptimizationSession(model, backend=backend) as session:
        source = iter(scenarios)
        index = 0
        while cancellation is None or not cancellation.cancelled:
            try:
                if tracker.remaining_s() is not None and tracker.remaining_s() <= 0: return
                item = next(source)
            except StopIteration:
                return
            name, updates = str(index), {}
            try:
                name, updates = item if isinstance(item, tuple) else (str(index), item)
                updates = dict(updates)
                controls=options
                if aggregate_budget is not None:
                    left=tracker.remaining_budget(); controls=options or SolveOptions()
                    if controls.budget is not None:
                        def intersect(a,b): return b if a is None else a if b is None else min(a,b)
                        left=replace(left,wall_time_s=intersect(left.wall_time_s,controls.budget.wall_time_s),
                                     memory_mb=intersect(left.memory_mb,controls.budget.memory_mb),threads=intersect(left.threads,controls.budget.threads))
                    controls=replace(controls,budget=left)
                result = session.solve(
                    updates={**baseline, **updates},
                    **({} if controls is None else {"options": controls}),
                )
                summary = summarize(result)
                pause=perf_counter()
                yield ScenarioResult(index, str(name), updates, summary, result,
                                     problem=session.problem_snapshot(),aggregate_budget_report=tracker.report())
                tracker.started+=perf_counter()-pause
            except Exception as exc:
                if on_error == "raise":
                    raise
                pause=perf_counter()
                yield ScenarioResult(
                    index, str(name), updates, error=f"{type(exc).__name__}: {exc}",aggregate_budget_report=tracker.report()
                )
                tracker.started+=perf_counter()-pause
            index += 1


def scenario_statistics(results):
    rows = list(results)
    feasible = [r for r in rows if r.summary is not None and r.summary.feasible]
    objectives = [r.summary.objective for r in feasible if r.summary.objective is not None]
    return {
        "count": len(rows),
        "feasible": len(feasible),
        "errors": sum(r.error is not None for r in rows),
        "minimum_objective": min(objectives, default=None),
        "maximum_objective": max(objectives, default=None),
    }
