from dataclasses import dataclass, field, replace, asdict
from importlib.util import find_spec
from threading import RLock
from time import perf_counter
import numpy as np
from solverpilot._immutability import readonly_array, deep_freeze
from solverpilot._synchronization import serialized
from solverpilot.exceptions import BackendUnavailableError
from solverpilot.problem import VariableDomain, ObjectiveSense
from solverpilot.validate import ValidationTolerances, ValidationReport
from solverpilot.runtime.result import OptimalityEvidence
from .problem import GlobalQuadraticProblem, FactorableProblem
from .expressions import to_scip
from .intervals import bound_node, validate_domains, upper_float, F
from .validation import validate_global_solution


@dataclass(frozen=True, slots=True)
class GlobalSolveResult:
    backend: str
    backend_status: str
    x: np.ndarray | None
    objective_reported: float | None
    validation: ValidationReport
    raw_statistics: dict
    problem_data_hash: str

    def __post_init__(self):
        if self.x is not None:
            object.__setattr__(self, "x", readonly_array(self.x, dtype=float))
        object.__setattr__(self, "raw_statistics", deep_freeze(self.raw_statistics))

    @property
    def status(self):
        return (
            self.backend_status if self.x is None or self.validation.valid else "invalid_solution"
        )

    @property
    def objective(self):
        return self.validation.objective_recomputed

    @property
    def optimality_evidence(self):
        return OptimalityEvidence(
            backend_reported_optimal=self.backend_status == "optimal",
            primal_validated=self.validation.valid,
        )


@dataclass(slots=True)
class SCIPGlobalBackend:
    """Numerical global SCIP path. Independent generic global proof is not claimed."""

    time_limit_s: float | None = 60.0
    threads: int = 1
    tolerance: float = 1e-9
    relative_gap: float = 0.0
    node_limit: int | None = None
    memory_mb: float | None = None
    verbose: bool = False
    _lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _active: bool = field(default=False, init=False, repr=False, compare=False)

    @property
    def name(self):
        return "scip-global"

    @property
    def compile_target(self):
        return "global"

    def is_available(self):
        return find_spec("pyscipopt") is not None

    @property
    def capability_manifest_v2(self):
        from .capabilities import global_manifest

        return global_manifest(self)

    def capabilities(self):
        if not self.is_available():
            return {"available": False, "exact_milp": False}
        import pyscipopt

        m = pyscipopt.Model()
        m.hideOutput()
        params = m.getParams()
        result = {
            "available": True,
            "native_version": ".".join(
                str(v) for v in (m.getMajorVersion(), m.getMinorVersion(), m.getTechVersion())
            ),
            "matrix_qp": True,
            "matrix_miqp": True,
            "bounded_factorable_minlp": True,
            "exact_milp": "exact/enable" in params,
            "certificate_output": "certificate/filename" in params,
            "independent_global_proof": False,
        }
        m.freeProb()
        return result

    @serialized
    def solve(self, problem, *, tolerances=None, progress=None, cancellation=None):
        if self._active:
            raise RuntimeError("a callback cannot reenter the same SCIP backend")
        if not isinstance(problem, (GlobalQuadraticProblem, FactorableProblem)):
            raise TypeError(
                "SCIPGlobalBackend requires GlobalQuadraticProblem or FactorableProblem"
            )
        if not self.is_available():
            raise BackendUnavailableError("install solverpilot[global] for global optimization")
        if type(self.threads) is not int or self.threads < 1:
            raise ValueError("threads must be a positive integer")
        for name, value in (("time_limit_s", self.time_limit_s), ("memory_mb", self.memory_mb)):
            if value is not None and (
                isinstance(value, bool) or not np.isfinite(value) or value <= 0
            ):
                raise ValueError(f"{name} must be finite and positive")
        if (
            isinstance(self.tolerance, bool)
            or not np.isfinite(self.tolerance)
            or not 1e-17 <= self.tolerance <= 1e-3
        ):
            raise ValueError("SCIP tolerance must lie in [1e-17, 1e-3]")
        if (
            isinstance(self.relative_gap, bool)
            or not np.isfinite(self.relative_gap)
            or self.relative_gap < 0
        ):
            raise ValueError("relative_gap must be finite and nonnegative")
        if self.node_limit is not None and (
            type(self.node_limit) is not int or self.node_limit < 0
        ):
            raise ValueError("node_limit must be a nonnegative integer")
        tol = tolerances or ValidationTolerances()
        if not isinstance(tol, ValidationTolerances):
            raise TypeError("tolerances must be ValidationTolerances")
        if cancellation is not None and cancellation.cancelled:
            return GlobalSolveResult(
                self.name,
                "cancelled",
                None,
                None,
                validate_global_solution(problem, None),
                {"backend_reported_optimal": False},
                problem.data_hash,
            )
        import pyscipopt as scip

        start = perf_counter()
        m = scip.Model("solverpilot-global")
        try:
            if not self.verbose:
                m.hideOutput()
            m.setIntParam("parallel/maxnthreads", self.threads)
            m.setRealParam("numerics/feastol", self.tolerance)
            m.setRealParam("limits/gap", self.relative_gap)
            if self.time_limit_s is not None:
                m.setRealParam("limits/time", float(self.time_limit_s))
            if self.memory_mb is not None:
                m.setRealParam("limits/memory", float(self.memory_mb))
            if self.node_limit is not None:
                m.setLongintParam("limits/nodes", self.node_limit)
            if isinstance(problem, GlobalQuadraticProblem):
                p = problem.linear
                lo = p.variable_lower
                hi = p.variable_upper
                domains = p.domains
                arrays = (
                    p.A.data,
                    p.c,
                    problem.P.data,
                    lo,
                    hi,
                    p.constraint_lower,
                    p.constraint_upper,
                    np.array([p.objective_offset]),
                )
            else:
                p = problem.relaxation
                lo = p.variable_lower
                hi = p.variable_upper
                domains = problem.domains
                arrays = (lo, hi, p.constraint_lower, p.constraint_upper)
            infinity = m.infinity()
            if any(np.any(np.abs(a[np.isfinite(a)]) >= infinity) for a in arrays):
                raise ValueError("finite model coefficient/bound reaches SCIP's infinity threshold")
            native_types = {
                VariableDomain.CONTINUOUS.value: "C",
                VariableDomain.INTEGER.value: "I",
                VariableDomain.BINARY.value: "B",
            }
            x = [
                m.addVar(
                    name=f"x[{i}]",
                    lb=float(a) if np.isfinite(a) else -infinity,
                    ub=float(b) if np.isfinite(b) else infinity,
                    vtype=native_types[d],
                )
                for i, (a, b, d) in enumerate(zip(lo, hi, domains))
            ]
            transformations = []

            def row(expr, lower, upper):
                expr = scip.quicksum([expr])
                if lower == upper:
                    m.addCons(expr == float(lower))
                else:
                    if np.isfinite(lower):
                        m.addCons(expr >= float(lower))
                    if np.isfinite(upper):
                        m.addCons(expr <= float(upper))

            if isinstance(problem, GlobalQuadraticProblem):
                for i in range(p.n_constraints):
                    begin, end = p.A.indptr[i : i + 2]
                    expr = scip.quicksum(
                        float(p.A.data[k]) * x[p.A.indices[k]] for k in range(begin, end)
                    )
                    row(expr, p.constraint_lower[i], p.constraint_upper[i])
                coo = problem.P.tocoo()
                objective = (
                    scip.quicksum(
                        0.5 * float(v) * x[int(i)] * x[int(j)]
                        for i, j, v in zip(coo.row, coo.col, coo.data)
                    )
                    + scip.quicksum(float(v) * x[i] for i, v in enumerate(p.c))
                    + float(p.objective_offset)
                )
            else:
                cache = {}
                pending = [
                    p.objective_node,
                    *(c.node for c in p.constraints),
                    *(i.constraint.node for i in problem.indicators),
                ]
                while pending:
                    node = pending.pop()
                    pending.extend(node.args)
                    value = (
                        node.payload
                        if node.kind == "constant"
                        else p.parameter_values[node.payload]
                        if node.kind == "parameter"
                        else None
                    )
                    if value is not None and np.any(np.abs(value) >= infinity):
                        raise ValueError("expression coefficient reaches SCIP's infinity threshold")
                for node in (
                    p.objective_node,
                    *(c.node for c in p.constraints),
                    *(i.constraint.node for i in problem.indicators),
                ):
                    validate_domains(problem, node)
                objective = to_scip(problem, p.objective_node, x, scip, cache).item()
                for c in p.constraints:
                    expr = to_scip(problem, c.node, x, scip, cache).reshape(-1)
                    for value, lower, upper in zip(expr, c.lower, c.upper):
                        row(value, lower, upper)
                for indicator in problem.indicators:
                    c = indicator.constraint
                    enclosure = bound_node(problem, c.node)
                    if enclosure is None:
                        raise ValueError("no exact interval bound for indicator body")
                    expr = to_scip(problem, c.node, x, scip, cache).reshape(-1)
                    active = (
                        x[indicator.variable_index]
                        if indicator.active_value
                        else 1 - x[indicator.variable_index]
                    )
                    for value, rng, lower, upper in zip(
                        expr, np.asarray(enclosure).reshape(-1), c.lower, c.upper
                    ):
                        for side, bound in (("upper", upper), ("lower", lower)):
                            if not np.isfinite(bound):
                                continue
                            exact = max(
                                F(0),
                                rng.hi - F(float(bound))
                                if side == "upper"
                                else F(float(bound)) - rng.lo,
                            )
                            M = upper_float(exact)
                            if M >= infinity:
                                raise ValueError("indicator M exceeds SCIP infinity threshold")
                            lhs = value - float(bound) if side == "upper" else float(bound) - value
                            m.addCons(scip.quicksum([lhs]) <= M * (1 - active))
                            transformations.append(
                                {
                                    "kind": "interval_big_m",
                                    "source_id": c.source_id,
                                    "side": side,
                                    "M": M,
                                    "exact_upper_bound": str(exact),
                                    "variable_index": indicator.variable_index,
                                    "active_value": indicator.active_value,
                                }
                            )
            eta = m.addVar(name="objective_epigraph", lb=-infinity, ub=infinity)
            minimize = problem.objective_sense is ObjectiveSense.MINIMIZE
            m.addCons(eta >= objective if minimize else eta <= objective)
            m.setObjective(eta, "minimize" if minimize else "maximize")
            errors = []
            if progress is not None or cancellation is not None:
                from solverpilot.runtime.options import ProgressEvent

                name = self.name

                class Events(scip.Eventhdlr):
                    def eventinit(self):
                        self.catch = (
                            scip.SCIP_EVENTTYPE.BESTSOLFOUND | scip.SCIP_EVENTTYPE.NODEFOCUSED
                        )
                        self.model.catchEvent(self.catch, self)

                    def eventexit(self):
                        self.model.dropEvent(self.catch, self)

                    def eventexec(self, event):
                        try:
                            stop = cancellation is not None and cancellation.cancelled
                            if progress is not None and not stop:
                                stop = bool(
                                    progress(
                                        ProgressEvent(
                                            name,
                                            "search",
                                            self.model.getNNodes(),
                                            self.model.getSolvingTime(),
                                        )
                                    )
                                )
                            if stop:
                                self.model.interruptSolve()
                        except BaseException as exc:
                            errors.append(exc)
                            self.model.interruptSolve()

                m.includeEventhdlr(Events(), "solverpilot-progress", "progress and cancellation")
            built = perf_counter()
            if self.time_limit_s is not None:
                remaining = self.time_limit_s - (built - start)
                if remaining <= 0:
                    return GlobalSolveResult(
                        self.name,
                        "timelimit",
                        None,
                        None,
                        validate_global_solution(problem, None),
                        {"reason": "model construction exhausted wall time"},
                        problem.data_hash,
                    )
                m.setRealParam("limits/time", remaining)
            self._active = True
            try:
                m.optimize()
            finally:
                self._active = False
            if errors:
                raise RuntimeError("SCIP progress callback failed") from errors[0]
            solved = perf_counter()
            status = str(m.getStatus())
            solution = m.getBestSol()
            candidate = (
                None if solution is None else np.array([m.getSolVal(solution, v) for v in x])
            )
            objective_value = None if solution is None else float(m.getSolObjVal(solution))
            validation = validate_global_solution(
                problem, candidate, objective_reported=objective_value, tolerances=tol
            )
            dual = float(m.getDualbound())
            gap = float(m.getGap())
            raw = {
                "backend_reported_optimal": status == "optimal",
                "backend_reported_global": status == "optimal",
                "independently_verified_optimal": False,
                "bound_origin": "scip",
                "solver_dual_bound": dual if np.isfinite(dual) and abs(dual) < infinity else None,
                "solver_gap": gap if np.isfinite(gap) and gap < infinity else None,
                "nodes": m.getNNodes(),
                "transformations": transformations,
                "native_version": ".".join(
                    str(v) for v in (m.getMajorVersion(), m.getMinorVersion(), m.getTechVersion())
                ),
                "effective_tolerances": asdict(tol),
                "solver_feasibility_tolerance": m.getParam("numerics/feastol"),
                "requested_relative_gap": self.relative_gap,
                "phase_timings": {
                    "backend_build_s": built - start,
                    "solve_s": solved - built,
                    "validate_s": perf_counter() - solved,
                },
                "reuse_report": {
                    "workspace": "not_used",
                    "primal_dual_start": "not_used",
                    "numeric_factorization": "unknown",
                    "reason": "fresh SCIP model",
                },
                "exact_mode_available": "exact/enable" in m.getParams(),
            }
            return GlobalSolveResult(
                self.name, status, candidate, objective_value, validation, raw, problem.data_hash
            )
        finally:
            m.freeProb()


def solve_global(
    problem,
    *,
    backend=None,
    options=None,
    budget=None,
    tolerances=None,
    progress=None,
    cancellation=None,
):
    from solverpilot.model import Model, CompiledModel
    from solverpilot.runtime.options import expand_options

    controls = {
        k: v
        for k, v in dict(
            options=options,
            budget=budget,
            tolerances=tolerances,
            progress=progress,
            cancellation=cancellation,
        ).items()
        if v is not None
    }
    controls = expand_options(controls)
    budget = controls.pop("budget", None)
    from solverpilot.plan import SolveBudget

    if budget is not None and not isinstance(budget, SolveBudget):
        raise TypeError("budget must be SolveBudget")
    started = perf_counter()
    backend = SCIPGlobalBackend() if backend is None or backend == "scip-global" else backend
    if not isinstance(backend, SCIPGlobalBackend):
        raise TypeError("global solve requires SCIPGlobalBackend")
    if budget is not None:
        changes = {
            k: v
            for k, v in dict(
                time_limit_s=budget.wall_time_s, threads=budget.threads, memory_mb=budget.memory_mb
            ).items()
            if v is not None
        }
        backend = replace(backend, **changes)
    if isinstance(problem, Model):
        from .compiler import compile_global_model

        problem = compile_global_model(problem)
    if isinstance(problem, CompiledModel):
        problem = problem.execution_ir
    if budget is not None and budget.wall_time_s is not None:
        remaining = budget.wall_time_s - (perf_counter() - started)
        if remaining <= 0:
            return GlobalSolveResult(
                backend.name,
                "timelimit",
                None,
                None,
                validate_global_solution(problem, None),
                {"reason": "compilation exhausted wall time"},
                problem.data_hash,
            )
        backend = replace(backend, time_limit_s=remaining)
    return backend.solve(problem, **controls)
