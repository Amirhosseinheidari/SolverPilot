"""A common result view that preserves specialized evidence and legacy APIs."""

from dataclasses import dataclass, field, replace
from typing import Any, Mapping
from uuid import uuid4
from solverpilot._immutability import deep_freeze, readonly_array
from .options import SolveOptions


@dataclass(frozen=True, slots=True)
class SolutionSummary:
    status: str
    objective: float | None
    feasible: bool
    optimality: str
    backend: str
    problem_data_hash: str | None
    x: Any = None
    assignment: Mapping | None = None
    run_id: str = field(default_factory=lambda: str(uuid4()))

    def __post_init__(self):
        if self.x is not None:
            object.__setattr__(self, "x", readonly_array(self.x))
        if self.assignment is not None:
            object.__setattr__(self, "assignment", deep_freeze(self.assignment))


def summarize(result: Any) -> SolutionSummary:
    validation = getattr(result, "validation", None)
    feasible = bool(getattr(validation, "valid", getattr(result, "validation_valid", False)))
    raw = getattr(result, "raw_statistics", None) or {}
    trust = getattr(result, "optimality_evidence", None)
    independent = bool(trust is not None and trust.independently_verified_optimal)
    reported = bool(
        getattr(trust, "backend_reported_optimal", False)
        or raw.get("backend_reported_optimal", False)
        or getattr(result, "globally_proven", False)
        or getattr(result, "optimality_proven", False)
    )
    local = bool(getattr(result, "local_optimal_candidate", False))
    evidence = (
        (
            "independent_numerical_bound"
            if independent
            else "solver_reported"
            if reported
            else "local_candidate"
            if local
            else "not_established"
        )
        if feasible
        else "not_established"
    )
    trace = getattr(result, "trace", None)
    status = getattr(result, "status", getattr(result, "backend_status", "unknown"))
    status = str(getattr(status, "value", status))
    objective = getattr(validation, "objective_recomputed", getattr(validation, "objective", None))
    if objective is None:
        objective = getattr(result, "objective", getattr(result, "objective_reported", None))
    return SolutionSummary(
        status,
        objective,
        feasible,
        evidence,
        trace.backend
        if trace is not None
        else getattr(result, "backend", getattr(result, "algorithm", "unknown")),
        trace.problem_data_hash
        if trace is not None
        else getattr(result, "problem_data_hash", None),
        getattr(result, "x", None),
        getattr(result, "assignment", None),
    )


def solve_any(
    problem: Any, *, backend: Any = None, options: SolveOptions | None = None
) -> tuple[SolutionSummary, Any]:
    """Solve Model, CompiledModel, canonical LP/QP/conic/NLP/MINLP or CP.

    Return (common summary, specialized result). Unsupported controls raise
    an explicit error; no resource or proof guarantee is silently invented.
    """
    from solverpilot.model import Model, CompiledModel
    from solverpilot.problem import LinearProblem, QuadraticProblem
    from solverpilot.conic import ConicProblem, solve_conic
    from solverpilot.nlp import NLPProblem, CasadiIpoptBackend
    from solverpilot.minlp import MINLPProblem, solve_outer_approximation
    from solverpilot.cp import CPModel, CPProblem, ReferenceCPBackend
    from .auto import solve
    from .options import expand_options
    from .specialized import nlp_controls, minlp_controls
    from .budgeting import configured_backend

    options = options or SolveOptions()
    kwargs = expand_options({"options": options})
    from solverpilot.globalopt import GlobalQuadraticProblem, FactorableProblem, solve_global
    if isinstance(problem, Model):
        target = "global" if getattr(backend, "compile_target", None) == "global" or backend == "scip-global" else None
        problem = problem.compile(target=target)
    if isinstance(problem, CompiledModel):
        result = (
            problem.solve(backend=backend, **kwargs)
            if backend is not None
            else problem.solve(**kwargs)
        )
    elif isinstance(problem, (GlobalQuadraticProblem, FactorableProblem)):
        result = solve_global(problem, backend=backend, **kwargs)
    elif isinstance(problem, (LinearProblem, QuadraticProblem)):
        from .specialized import core_controls

        result = solve(problem, backend=backend, **core_controls(kwargs))
    elif isinstance(problem, ConicProblem):
        result = solve_conic(problem, backend=backend, **kwargs)
    elif isinstance(problem, NLPProblem):
        chosen, controls = nlp_controls(backend or CasadiIpoptBackend(), kwargs)
        result = chosen.solve(problem, **controls)
    elif isinstance(problem, MINLPProblem):
        controls = minlp_controls(kwargs)
        if backend is not None:
            controls["nlp_backend"] = backend
        result = replace(
            solve_outer_approximation(problem, **controls), problem_data_hash=problem.data_hash
        )
    else:
        if isinstance(problem, CPModel):
            problem = problem.compile()
        if not isinstance(problem, CPProblem):
            raise TypeError("unsupported problem type")
        from solverpilot.exceptions import BudgetNotSupportedError

        if options.progress is not None or options.cancellation is not None:
            raise BudgetNotSupportedError(
                "CP progress/cancellation is not available in the common API"
            )
        chosen = backend or ReferenceCPBackend()
        if isinstance(chosen, str):
            from .catalog import backend_catalog

            chosen = backend_catalog()[chosen]
        if options.budget is not None:
            budget = options.budget
            if budget.memory_mb is not None or not hasattr(chosen, "max_time_s"):
                raise BudgetNotSupportedError("this CP backend does not support requested budget")
            changes = {}
            if budget.wall_time_s is not None:
                changes["max_time_s"] = budget.wall_time_s
            if budget.threads is not None:
                changes["num_workers"] = budget.threads
            chosen = configured_backend(chosen, changes)
        result = chosen.solve(problem)
    summary = summarize(result)
    if summary.problem_data_hash is None and hasattr(problem, "data_hash"):
        summary = replace(summary, problem_data_hash=problem.data_hash)
    if isinstance(problem, CompiledModel) and summary.x is not None:
        summary = replace(summary, x=problem.reconstruct_primal(summary.x))
    return summary, result


class UnverifiedSolutionError(RuntimeError):
    """A checked solve retained a candidate without sufficient optimality evidence."""

    def __init__(self, summary: SolutionSummary, result: Any):
        super().__init__("independent numerical optimality bound was not established")
        self.summary, self.result = summary, result


def solve_verified(
    problem: Any, *, backend: Any = None, options: SolveOptions | None = None
) -> tuple[SolutionSummary, Any]:
    """Require a validated primal and an independent, domain-corrected bound.

    Failure retains the result on UnverifiedSolutionError for inspection. This
    Supported LP/QP and scoped SOC/RSOC/PSD witnesses are tolerance-qualified;
    this is not a proof about unrounded input or generic global nonlinear models.
    """
    summary, result = solve_any(problem, backend=backend, options=options)
    if not summary.feasible or summary.optimality != "independent_numerical_bound":
        raise UnverifiedSolutionError(summary, result)
    return summary, result
