"""Independent parameter scenarios using an owned reusable model session."""

from dataclasses import dataclass
from solverpilot._immutability import deep_freeze
from solverpilot.runtime.unified import summarize
from solverpilot.session import ReoptimizationSession


@dataclass(frozen=True, slots=True)
class ScenarioResult:
    index: int
    name: str
    updates: object
    summary: object = None
    result: object = None
    error: str | None = None

    def __post_init__(self):
        object.__setattr__(self, "updates", deep_freeze(self.updates))


def scenario_sweep(
    model, scenarios, *, backend=None, options=None, cancellation=None, on_error="raise"
):
    """Yield scenarios; unspecified parameters reset to the original baseline.

    Each input is an update mapping or (name, mapping). Input is consumed lazily.
    Close the iterator if stopping early, to release its private session.
    """
    if on_error not in ("raise", "record"):
        raise ValueError("on_error must be raise or record")
    baseline = {p.name: p.value for p in model.parameters}
    with ReoptimizationSession(model, backend=backend) as session:
        source = iter(scenarios)
        index = 0
        while cancellation is None or not cancellation.cancelled:
            try:
                item = next(source)
            except StopIteration:
                return
            name, updates = item if isinstance(item, tuple) else (str(index), item)
            updates = dict(updates)
            try:
                result = session.solve(
                    updates={**baseline, **updates},
                    **({} if options is None else {"options": options}),
                )
                summary = summarize(result)
                yield ScenarioResult(index, str(name), updates, summary, result)
            except Exception as exc:
                if on_error == "raise":
                    raise
                yield ScenarioResult(
                    index, str(name), updates, error=f"{type(exc).__name__}: {exc}"
                )
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
