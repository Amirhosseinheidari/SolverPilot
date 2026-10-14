from __future__ import annotations

import importlib.util
from dataclasses import dataclass, field
from importlib.metadata import PackageNotFoundError
from threading import RLock
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot._synchronization import serialized
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.hashing import hash_sparse_matrix
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError
from .metadata import version


def _highs_status_from_strings(model_status: str, primal_status: str) -> str:
    model = model_status.strip().lower()
    primal_feasible = "feasible" in primal_status.strip().lower() and "infeasible" not in primal_status.strip().lower()
    if "optimal" in model:
        return "optimal"
    if "unbounded or infeasible" in model or "infeasible or unbounded" in model:
        return "infeasible_or_unbounded"
    if "infeasible" in model:
        return "infeasible"
    if "unbounded" in model:
        return "unbounded"
    if any(token in model for token in ("time limit", "iteration limit", "solution limit", "objective bound", "objective target", "interrupt")):
        return "limit_feasible" if primal_feasible else "limit_no_solution"
    if any(token in model for token in ("model error", "solve error", "memory limit")):
        return "solver_error"
    return "unknown"


def _linear_csc(problem: LinearProblem) -> sparse.csc_matrix:
    A = problem.A.tocsc(copy=True)
    A.sort_indices()
    return A


def _lower_hessian_csc(problem: QuadraticProblem) -> sparse.csc_matrix:
    P = sparse.tril(problem.P, format="csc")
    P.sort_indices()
    return P


@dataclass(slots=True)
class HighspyNativeBackend:
    """Optional native HiGHS backend.

    Default solves are stateless. ``reuse=True, solver="simplex"`` retains a
    continuous LP only while its matrix, domains, objective sense and structural
    options are unchanged. Costs, row/column bounds and objective offset may be
    updated. Other changes rebuild; QP/MILP and automatic solver selection never
    retain state. Native basis validity is observed, not factorization reuse.

    Solve/reset/close operations are serialized. Do not mutate public options
    concurrently with a solve; runtime call-local budgets use the same lock.
    Per-call time limits may change without rebuilding. ``close`` is terminal; ``reset`` drops
    native state while leaving an open instance usable.
    """

    time_limit_s: float | None = None
    threads: int | None = None
    presolve: bool | None = None
    solver: str | None = None
    reuse: bool = field(default=False, kw_only=True)
    _lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _highs: object | None = field(default=None, init=False, repr=False, compare=False)
    _previous: LinearProblem | None = field(default=None, init=False, repr=False, compare=False)
    _matrix_hash: str | None = field(default=None, init=False, repr=False, compare=False)
    _configuration: tuple | None = field(default=None, init=False, repr=False, compare=False)
    _pending_reason: str = field(default="first_solve", init=False, repr=False, compare=False)
    _closed: bool = field(default=False, init=False, repr=False, compare=False)
    _solving: bool = field(default=False, init=False, repr=False, compare=False)
    _scheduler_conflict: bool = field(default=False, init=False, repr=False, compare=False)
    _native_callback: object | None = field(default=None, init=False, repr=False, compare=False)
    _allow_thread_isolation: bool = field(default=True, init=False, repr=False, compare=False)

    def __post_init__(self):
        if not isinstance(self.reuse, bool):
            raise TypeError("reuse must be a boolean")

    @property
    def _execution_lock(self):
        """Allow the executor to own snapshots within the reentrant boundary."""
        return self._lock

    @property
    def manifest(self) -> BackendManifest:
        pkg_version = None
        if self.is_available():
            try:
                pkg_version = version("highspy")
            except PackageNotFoundError:
                pass
        return BackendManifest(
            name="highspy-native",
            version=pkg_version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.NATIVE,
                # No explicit basis/start injection or arbitrary matrix update API.
                Capability.MIP_START: SupportLevel.UNKNOWN,
                Capability.PRIMAL_START: SupportLevel.UNKNOWN,
                Capability.DUAL_START: SupportLevel.UNKNOWN,
                Capability.BASIS_START: SupportLevel.UNKNOWN,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNKNOWN,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNKNOWN,
                Capability.NATIVE_REOPTIMIZATION: (
                    SupportLevel.NATIVE if self.reuse and self.solver == "simplex"
                    else SupportLevel.UNSUPPORTED),
                Capability.CALLBACK_PROGRESS: SupportLevel.UNKNOWN,
                Capability.IIS: SupportLevel.UNKNOWN,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
            },
            metadata={
                "native_adapter": True,
                "stateful": bool(self.reuse and self.solver == "simplex"),
                "reuse_claim": "opt-in fixed-matrix continuous simplex LP vector updates",
                "reuse_enabled": bool(self.reuse and self.solver == "simplex"),
                "reuse_scope": "costs, row bounds, column bounds, offset; identical A, domains, sense, threads, presolve and solver",
                "time_limit_policy": "mutable per-call allowance; cumulative native clock offset; removing limit restores infinity",
                "matrix_update_policy": "rebuild even when only matrix values change",
                "basis_reuse_evidence": "validity observations only; no factorization claim",
                "thread_conflict_policy": "isolated cold execution; never reset the global scheduler",
            },
        )

    def is_available(self) -> bool:
        if importlib.util.find_spec("highspy") is None:
            return False
        try:
            # Finding package metadata is insufficient: the extension may not
            # load, or another HiGHS ABI may already occupy its library name.
            # Resolve the dedicated native adapter before verification bridges
            # are examined by the default registry.
            importlib.import_module("highspy")
        except (ImportError, OSError):
            return False
        return True

    def _discard(self, reason):
        # Dropping the owned handle releases native state without touching the
        # process-global HiGHS scheduler or other backend instances.
        self._highs = None
        self._previous = None
        self._matrix_hash = None
        self._configuration = None
        self._native_callback = None
        self._scheduler_conflict = False
        self._pending_reason = reason

    @serialized
    def reset(self):
        if self._closed:
            raise RuntimeError("HiGHS backend is closed")
        if self._solving:
            raise RuntimeError("cannot reset an active HiGHS solve")
        self._discard("explicit_reset")

    @serialized
    def close(self):
        if self._solving:
            raise RuntimeError("cannot close an active HiGHS solve")
        self._discard("closed")
        self._closed = True

    @serialized
    def __enter__(self):
        if self._closed:
            raise RuntimeError("HiGHS backend is closed")
        return self

    def __exit__(self, *_):
        self.close()

    @serialized
    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if self._closed:
            raise RuntimeError("HiGHS backend is closed")
        if self._solving:
            raise RuntimeError("a HiGHS solve cannot reenter its backend")
        self._solving = True
        try:
            return self._solve(problem)
        except BaseException:
            # No partially updated or failed native object survives an error.
            self._discard("previous_solve_failed")
            raise
        finally:
            self._solving = False

    @staticmethod
    def _require_ok(highspy, status, operation):
        if status != highspy.HighsStatus.kOk:
            raise RuntimeError(f"HiGHS rejected {operation}: {status}")

    def _options(self):
        if not isinstance(self.reuse, bool):
            raise TypeError("reuse must be a boolean")
        options = {"output_flag": False}
        if self.time_limit_s is not None:
            if isinstance(self.time_limit_s, bool) or not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be finite and positive")
            options["time_limit"] = float(self.time_limit_s)
        if self.threads is not None:
            if type(self.threads) is not int or self.threads <= 0:
                raise ValueError("threads must be a positive integer")
            options["threads"] = self.threads
        if self.presolve is not None:
            options["presolve"] = "on" if self.presolve else "off"
        if self.solver is not None:
            options["solver"] = str(self.solver)
        return options

    def _build_native(self, highspy, problem, options):
        h = highspy.Highs()
        for key, value in options.items():
            self._require_ok(highspy, h.setOptionValue(key, value), f"option {key}")
        model = (self._make_qp_model(highspy, problem) if isinstance(problem, QuadraticProblem)
                 else self._make_lp_model(highspy, problem))
        if h.passModel(model) == highspy.HighsStatus.kError:
            raise RuntimeError("HiGHS rejected model")
        if self.threads is not None and self._allow_thread_isolation:
            def observe(_kind, message, _output, _input, _data):
                if 'global scheduler has already been initialized' in message and 'threads' in message:
                    self._scheduler_conflict = True
            self._native_callback = observe
            self._require_ok(highspy, h.setCallback(observe, None), 'scheduler diagnostic callback')
            self._require_ok(highspy, h.startCallback(highspy.cb.kCallbackLogging), 'scheduler diagnostic logging')
            self._require_ok(highspy, h.setOptionValue('log_to_console', False), 'silent diagnostic logging')
            self._require_ok(highspy, h.setOptionValue('output_flag', True), 'scheduler diagnostic output')
        return h

    def _update_native(self, highspy, h, problem):
        previous = self._previous
        changed = []
        if not np.array_equal(previous.c, problem.c):
            indices = np.arange(problem.n_variables, dtype=np.int32)
            self._require_ok(highspy, h.changeColsCost(problem.n_variables, indices, problem.c), "column costs")
            changed.append("costs")
        if (not np.array_equal(previous.variable_lower, problem.variable_lower)
                or not np.array_equal(previous.variable_upper, problem.variable_upper)):
            indices = np.arange(problem.n_variables, dtype=np.int32)
            self._require_ok(highspy, h.changeColsBounds(problem.n_variables, indices,
                problem.variable_lower, problem.variable_upper), "column bounds")
            changed.append("column_bounds")
        if (not np.array_equal(previous.constraint_lower, problem.constraint_lower)
                or not np.array_equal(previous.constraint_upper, problem.constraint_upper)):
            indices = np.arange(problem.n_constraints, dtype=np.int32)
            self._require_ok(highspy, h.changeRowsBounds(problem.n_constraints, indices,
                problem.constraint_lower, problem.constraint_upper), "row bounds")
            changed.append("row_bounds")
        if previous.objective_offset != problem.objective_offset:
            self._require_ok(highspy, h.changeObjectiveOffset(problem.objective_offset), "objective offset")
            changed.append("objective_offset")
        return changed

    def _solve(self, problem):
        if not self.is_available():
            raise BackendUnavailableError("highspy is not installed")
        import highspy

        prepare_start = perf_counter()
        options = self._options()
        # Runtime budgets subtract per-call setup time. These varying remaining
        # allowances do not invalidate a compatible LP or its native basis.
        configuration = tuple(sorted((key, value) for key, value in options.items() if key != "time_limit"))
        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
        elif not isinstance(problem, LinearProblem):
            raise TypeError(f"unsupported problem type: {type(problem)!r}")
        eligible = (self.reuse and self.solver == "simplex" and isinstance(problem, LinearProblem)
                    and not problem.has_integer_variables)
        matrix_hash = hash_sparse_matrix(problem.A) if eligible else None
        had_workspace = self._highs is not None
        reason = "unchanged_problem"
        if not self.reuse:
            reason = "reuse_disabled"
        elif not isinstance(problem, LinearProblem) or problem.has_integer_variables:
            reason = "nonreusable_problem"
        elif self.solver != "simplex":
            reason = "explicit_simplex_required"
        elif not had_workspace:
            reason = self._pending_reason
        elif configuration != self._configuration:
            reason = "configuration_changed"
        elif problem.objective_sense != self._previous.objective_sense:
            reason = "objective_sense_changed"
        elif problem.structural_hash != self._previous.structural_hash:
            reason = "structure_or_domains_changed"
        elif matrix_hash != self._matrix_hash:
            reason = "matrix_values_changed"

        applied = bool(eligible and had_workspace and reason == "unchanged_problem")
        before_update = None
        changed = []
        fallback_error = None
        update_s = 0.0
        build_start = prepare_start
        if applied:
            h = self._highs
            before_update = bool(h.getBasis().valid)
            try:
                changed = self._update_native(highspy, h, problem)
                reason = "fixed_matrix_vector_update" if changed else "unchanged_problem"
            except Exception as exc:  # noqa: BLE001 -- any partial native update requires a fresh model
                fallback_error = f"{type(exc).__name__}: {exc}"
                reason = "native_update_failed"
                applied = False
                self._discard(reason)
                build_start = perf_counter()
                update_s = build_start - prepare_start
        if not applied:
            self._discard(reason)
            h = self._build_native(highspy, problem, options)

        # HiGHS's native run clock accumulates across calls. Give each retained
        # run its requested allowance, rather than reusing an exhausted limit.
        runtime_before = float(h.getRunTime())
        effective_time_limit = None
        if "time_limit" in options:
            effective_time_limit = runtime_before + options["time_limit"]
            if not np.isfinite(effective_time_limit):
                raise ValueError("cumulative native time limit overflow")
            self._require_ok(highspy, h.setOptionValue("time_limit", effective_time_limit), "per-run time limit")
        else:
            # The handle may retain a previous call's finite native option.
            self._require_ok(highspy, h.setOptionValue("time_limit", highspy.kHighsInf), "unlimited run time")
        basis_before_run = bool(h.getBasis().valid)
        prepared = perf_counter()
        build_s = 0.0 if applied else prepared - build_start
        if applied:
            update_s = prepared - prepare_start
        self._scheduler_conflict = False
        if h.run() == highspy.HighsStatus.kError:
            if self._scheduler_conflict and self._allow_thread_isolation:
                from ._highs_thread_transport import solve_isolated_threads
                self._discard('isolated_thread_configuration')
                return solve_isolated_threads(problem, self, started=prepare_start)
            raise RuntimeError("HiGHS run failed; native state discarded")
        solved = perf_counter()
        info = h.getInfo()
        model_status = h.modelStatusToString(h.getModelStatus())
        primal_status = h.solutionStatusToString(info.primal_solution_status)
        backend_status = _highs_status_from_strings(model_status, primal_status)

        x = None
        if backend_status in {"optimal", "limit_feasible"}:
            solution = h.getSolution()
            x_candidate = np.asarray(list(solution.col_value), dtype=np.float64)
            if (solution.value_valid and x_candidate.shape == (problem.n_variables,)
                    and np.all(np.isfinite(x_candidate))):
                x = x_candidate

        objective = None
        objective_internal = None
        if x is not None:
            try:
                candidate = float(h.getObjectiveValue())
                if np.isfinite(candidate):
                    objective_internal = candidate
                    objective = candidate
            except (RuntimeError, TypeError, ValueError, OverflowError):
                objective_internal = None

        linear = problem.linear if isinstance(problem, QuadraticProblem) else problem
        dual = None
        if x is not None and not linear.has_integer_variables:
            solution = h.getSolution()
            if solution.dual_valid:
                sign = 1. if linear.objective_sense is ObjectiveSense.MINIMIZE else -1.
                dual = (-sign*np.r_[solution.row_dual, solution.col_dual]).tolist()
        raw = {
            "canonical_dual": dual,
            "phase_timings": {"backend_build_s": build_s, "backend_update_s": update_s,
                              "solve_s": solved-prepared},
            "model_status": str(model_status),
            "primal_solution_status": str(primal_status),
            "simplex_iteration_count": int(getattr(info, "simplex_iteration_count", 0)),
            "ipm_iteration_count": int(getattr(info, "ipm_iteration_count", 0)),
            "qp_iteration_count": int(getattr(info, "qp_iteration_count", 0)),
            "mip_node_count": int(getattr(info, "mip_node_count", 0)),
            "reuse_applied": applied,
            "reuse_mode": ("native_lp_update" if changed else "native_lp_resolve") if applied else (
                "cold_rebuild" if had_workspace or reason in {"explicit_reset", "previous_solve_failed"}
                else "cold_native_solve"),
            "reuse_reason": reason,
            "updated_fields": changed if applied else [],
            "reuse_fallback_error": fallback_error,
            "reuse_observations": {"native_model_retained": applied,
                "basis_valid_before_update": before_update,
                "basis_valid_before_run": basis_before_run,
                "basis_valid_after_run": bool(h.getBasis().valid),
                "factorization_reuse": "not_observed"},
            "native_time_limit_s": effective_time_limit,
            "requested_time_limit_s": options.get("time_limit"),
            "native_runtime_before_s": runtime_before,
            "native_runtime_after_s": float(h.getRunTime()),
            "objective_internal": objective_internal,
        }
        if isinstance(problem, LinearProblem) and not problem.has_integer_variables:
            basis = h.getBasis()
            if basis.valid:
                raw['lp_basis'] = {
                    'basic_columns': [j for j, s in enumerate(basis.col_status)
                                      if s == highspy.HighsBasisStatus.kBasic],
                    'nonbasic_rows': [i for i, s in enumerate(basis.row_status)
                                     if s != highspy.HighsBasisStatus.kBasic],
                    'problem_hash': problem.data_hash,
                    'scope': 'numerical elimination hint; not a certificate',
                }
        if eligible and backend_status not in {"solver_error", "unknown"}:
            self._highs, self._previous = h, problem
            self._configuration, self._matrix_hash = configuration, matrix_hash
        else:
            self._discard("previous_solve_not_reusable")
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective,
            raw_statistics=raw,
        )

    @staticmethod
    def _make_lp_model(highspy, problem: LinearProblem):
        lp = highspy.HighsLp()
        lp.num_col_ = problem.n_variables
        lp.num_row_ = problem.n_constraints
        lp.col_cost_ = np.asarray(problem.c, dtype=np.float64)
        lp.col_lower_ = np.asarray(problem.variable_lower, dtype=np.float64)
        lp.col_upper_ = np.asarray(problem.variable_upper, dtype=np.float64)
        lp.row_lower_ = np.asarray(problem.constraint_lower, dtype=np.float64)
        lp.row_upper_ = np.asarray(problem.constraint_upper, dtype=np.float64)
        lp.sense_ = highspy.ObjSense.kMinimize if problem.objective_sense is ObjectiveSense.MINIMIZE else highspy.ObjSense.kMaximize
        lp.offset_ = float(problem.objective_offset)
        A = _linear_csc(problem)
        lp.a_matrix_.format_ = highspy.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.asarray(A.indptr, dtype=np.int64)
        lp.a_matrix_.index_ = np.asarray(A.indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(A.data, dtype=np.float64)
        if problem.has_integer_variables:
            lp.integrality_ = [
                highspy.HighsVarType.kInteger
                if domain in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}
                else highspy.HighsVarType.kContinuous
                for domain in problem.domains
            ]
        return lp

    @classmethod
    def _make_qp_model(cls, highspy, problem: QuadraticProblem):
        model = highspy.HighsModel()
        model.lp_ = cls._make_lp_model(highspy, problem.linear)
        P = _lower_hessian_csc(problem)
        model.hessian_.dim_ = problem.n_variables
        model.hessian_.format_ = highspy.HessianFormat.kTriangular
        model.hessian_.start_ = np.asarray(P.indptr, dtype=np.int64)
        model.hessian_.index_ = np.asarray(P.indices, dtype=np.int32)
        model.hessian_.value_ = np.asarray(P.data, dtype=np.float64)
        return model
