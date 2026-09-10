from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


def _available() -> bool:
    return importlib.util.find_spec("scipy.optimize._highspy._core") is not None


def _status_name(core, status) -> str:
    mapping = {
        core.HighsModelStatus.kOptimal: "optimal",
        core.HighsModelStatus.kInfeasible: "infeasible",
        core.HighsModelStatus.kUnbounded: "unbounded",
        core.HighsModelStatus.kUnboundedOrInfeasible: "infeasible_or_unbounded",
        core.HighsModelStatus.kTimeLimit: "limit_no_solution",
        core.HighsModelStatus.kIterationLimit: "limit_no_solution",
        core.HighsModelStatus.kSolutionLimit: "limit_no_solution",
        core.HighsModelStatus.kObjectiveBound: "limit_no_solution",
        core.HighsModelStatus.kObjectiveTarget: "limit_no_solution",
        core.HighsModelStatus.kInterrupt: "limit_no_solution",
        core.HighsModelStatus.kModelError: "solver_error",
        core.HighsModelStatus.kSolveError: "solver_error",
        core.HighsModelStatus.kPresolveError: "solver_error",
        core.HighsModelStatus.kPostsolveError: "solver_error",
        core.HighsModelStatus.kLoadError: "solver_error",
        core.HighsModelStatus.kMemoryLimit: "solver_error",
    }
    return mapping.get(status, "unknown")


def _ok(core, status) -> bool:
    return status in {core.HighsStatus.kOk, core.HighsStatus.kWarning}


def _linear_csc(problem: LinearProblem) -> sparse.csc_matrix:
    A = problem.A.tocsc(copy=True)
    A.sort_indices()
    return A


def _lower_hessian_csc(problem: QuadraticProblem) -> sparse.csc_matrix:
    P = sparse.tril(problem.P, format="csc")
    P.sort_indices()
    return P




@dataclass(frozen=True, slots=True)
class HighsDevIISResult:
    valid: bool
    row_indices: tuple[int, ...]
    row_bound_statuses: tuple[int, ...]
    col_indices: tuple[int, ...]
    col_bound_statuses: tuple[int, ...]
    strategy: int

@dataclass(slots=True)
class ScipyVendoredHighsDevBackend:
    """Stateful development-only HiGHS backend using SciPy's private vendored bindings.

    This class exists only to *verify architecture and reuse semantics* in environments
    where the public ``highspy`` package cannot be installed.  It must never be treated
    as a production dependency: ``scipy.optimize._highspy`` is private API and may change
    without notice.

    For continuous LPs with unchanged structure, the backend updates the existing model,
    restores the previous valid basis explicitly, and reruns HiGHS.  For MILPs with
    unchanged structure it injects the previous incumbent as a solution start.  Structural
    changes force a rebuild.  QPs are solved natively but are rebuilt in this M3 verifier.
    """

    time_limit_s: float | None = None
    threads: int | None = None
    solver: str | None = None
    presolve: bool | None = None
    _highs: object | None = field(default=None, init=False, repr=False)
    _last_problem: LinearProblem | QuadraticProblem | None = field(default=None, init=False, repr=False)
    _last_x: np.ndarray | None = field(default=None, init=False, repr=False)

    @property
    def manifest(self) -> BackendManifest:
        version = None
        if self.is_available():
            from scipy.optimize._highspy import _core as core
            version = f"{core.HIGHS_VERSION_MAJOR}.{core.HIGHS_VERSION_MINOR}.{core.HIGHS_VERSION_PATCH}-scipy-vendored"
        return BackendManifest(
            name="scipy-vendored-highs-native-dev",
            version=version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.NATIVE,
                Capability.MIP_START: SupportLevel.NATIVE,
                Capability.PRIMAL_START: SupportLevel.NATIVE,
                Capability.DUAL_START: SupportLevel.UNSUPPORTED,
                Capability.BASIS_START: SupportLevel.NATIVE,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.NATIVE,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNKNOWN,
                Capability.IIS: SupportLevel.NATIVE,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
            },
            metadata={
                "native_adapter": True,
                "stateful": True,
                "development_only": True,
                "private_scipy_api": True,
                "do_not_register_by_default": True,
            },
        )

    def is_available(self) -> bool:
        return _available()

    def _new_highs(self):
        if not self.is_available():
            raise BackendUnavailableError("SciPy vendored HiGHS bindings are unavailable")
        from scipy.optimize._highspy import _core as core
        h = core._Highs()
        h.setOptionValue("output_flag", False)
        if self.time_limit_s is not None:
            if self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            h.setOptionValue("time_limit", float(self.time_limit_s))
        if self.threads is not None:
            if self.threads <= 0:
                raise ValueError("threads must be positive")
            h.setOptionValue("threads", int(self.threads))
        if self.solver is not None:
            h.setOptionValue("solver", str(self.solver))
        if self.presolve is not None:
            h.setOptionValue("presolve", "on" if self.presolve else "off")
        return core, h

    @staticmethod
    def _make_lp(core, problem: LinearProblem):
        lp = core.HighsLp()
        lp.num_col_ = problem.n_variables
        lp.num_row_ = problem.n_constraints
        lp.col_cost_ = np.asarray(problem.c, dtype=np.float64)
        lp.col_lower_ = np.asarray(problem.variable_lower, dtype=np.float64)
        lp.col_upper_ = np.asarray(problem.variable_upper, dtype=np.float64)
        lp.row_lower_ = np.asarray(problem.constraint_lower, dtype=np.float64)
        lp.row_upper_ = np.asarray(problem.constraint_upper, dtype=np.float64)
        lp.sense_ = core.ObjSense.kMinimize if problem.objective_sense is ObjectiveSense.MINIMIZE else core.ObjSense.kMaximize
        lp.offset_ = float(problem.objective_offset)
        A = _linear_csc(problem)
        lp.a_matrix_.format_ = core.MatrixFormat.kColwise
        lp.a_matrix_.start_ = np.asarray(A.indptr, dtype=np.int32)
        lp.a_matrix_.index_ = np.asarray(A.indices, dtype=np.int32)
        lp.a_matrix_.value_ = np.asarray(A.data, dtype=np.float64)
        if problem.has_integer_variables:
            lp.integrality_ = [
                core.HighsVarType.kInteger
                if domain in {VariableDomain.INTEGER.value, VariableDomain.BINARY.value}
                else core.HighsVarType.kContinuous
                for domain in problem.domains
            ]
        return lp

    @classmethod
    def _make_model(cls, core, problem: LinearProblem | QuadraticProblem):
        if isinstance(problem, QuadraticProblem):
            model = core.HighsModel()
            model.lp_ = cls._make_lp(core, problem.linear)
            P = _lower_hessian_csc(problem)
            model.hessian_.dim_ = problem.n_variables
            model.hessian_.format_ = core.HessianFormat.kTriangular
            model.hessian_.start_ = np.asarray(P.indptr, dtype=np.int32)
            model.hessian_.index_ = np.asarray(P.indices, dtype=np.int32)
            model.hessian_.value_ = np.asarray(P.data, dtype=np.float64)
            return model
        return cls._make_lp(core, problem)

    def _rebuild(self, problem: LinearProblem | QuadraticProblem):
        core, h = self._new_highs()
        status = h.passModel(self._make_model(core, problem))
        if not _ok(core, status):
            raise RuntimeError(f"vendored HiGHS rejected model: {status}")
        self._highs = h
        return core, h

    @staticmethod
    def _apply_same_structure_lp_updates(core, h, old: LinearProblem, new: LinearProblem) -> None:
        n = new.n_variables
        idx = np.arange(n, dtype=np.int32)
        status = h.changeColsCost(n, idx, np.asarray(new.c, dtype=np.float64))
        if not _ok(core, status):
            raise RuntimeError(f"HiGHS cost update failed: {status}")
        status = h.changeColsBounds(
            n,
            idx,
            np.asarray(new.variable_lower, dtype=np.float64),
            np.asarray(new.variable_upper, dtype=np.float64),
        )
        if not _ok(core, status):
            raise RuntimeError(f"HiGHS variable-bound update failed: {status}")
        for i in range(new.n_constraints):
            status = h.changeRowBounds(i, float(new.constraint_lower[i]), float(new.constraint_upper[i]))
            if not _ok(core, status):
                raise RuntimeError(f"HiGHS row-bound update failed at row {i}: {status}")
        status = h.changeObjectiveOffset(float(new.objective_offset))
        if not _ok(core, status):
            raise RuntimeError(f"HiGHS objective-offset update failed: {status}")
        sense = core.ObjSense.kMinimize if new.objective_sense is ObjectiveSense.MINIMIZE else core.ObjSense.kMaximize
        status = h.changeObjectiveSense(sense)
        if not _ok(core, status):
            raise RuntimeError(f"HiGHS objective-sense update failed: {status}")

        # Structural hash equality guarantees identical A sparsity and variable domains.
        # Same structural hash means CSR indptr/indices are identical.  Compare the
        # contiguous data arrays directly so the common "A unchanged" path is O(nnz)
        # vectorized rather than O(nnz) Python scalar indexing.
        changed = np.flatnonzero(old.A.data != new.A.data)
        if changed.size:
            rows = np.searchsorted(new.A.indptr[1:], changed, side="right")
            cols = new.A.indices[changed]
            vals = new.A.data[changed]
            for i, j, value in zip(rows, cols, vals, strict=True):
                status = h.changeCoeff(int(i), int(j), float(value))
                if not _ok(core, status):
                    raise RuntimeError(f"HiGHS coefficient update failed at ({i}, {j}): {status}")

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
        if not self.is_available():
            raise BackendUnavailableError("SciPy vendored HiGHS bindings are unavailable")
        from scipy.optimize._highspy import _core as core

        reuse_applied = False
        reuse_mode = "cold_rebuild"
        old = self._last_problem
        same_structure = (
            old is not None
            and type(old) is type(problem)
            and old.structural_hash == problem.structural_hash
        )

        # QP data mutation is deliberately rebuilt in the M3 verifier because this private
        # binding does not expose a stable Hessian-update contract we are willing to claim.
        can_update_lp = same_structure and isinstance(old, LinearProblem) and isinstance(problem, LinearProblem)

        if can_update_lp and self._highs is not None:
            h = self._highs
            basis = None
            if not problem.has_integer_variables:
                candidate = h.getBasis()
                if bool(getattr(candidate, "valid", False)):
                    basis = candidate
            self._apply_same_structure_lp_updates(core, h, old, problem)
            if problem.has_integer_variables and self._last_x is not None:
                indices = np.arange(problem.n_variables, dtype=np.int32)
                status = h.setSolution(problem.n_variables, indices, np.asarray(self._last_x, dtype=np.float64))
                if _ok(core, status):
                    reuse_applied = True
                    reuse_mode = "mip_solution_start"
            elif basis is not None:
                status = h.setBasis(basis)
                if _ok(core, status):
                    reuse_applied = True
                    reuse_mode = "explicit_basis_hot_start"
        else:
            core, h = self._rebuild(problem)

        runtime_before = float(h.getRunTime())
        run_status = h.run()
        runtime_after = float(h.getRunTime())
        if not _ok(core, run_status):
            raise RuntimeError(f"HiGHS run failed: {run_status}")

        model_status_enum = h.getModelStatus()
        backend_status = _status_name(core, model_status_enum)
        info = h.getInfo()
        primal_status = h.solutionStatusToString(info.primal_solution_status)
        if backend_status.startswith("limit") and "feasible" in primal_status.lower() and "infeasible" not in primal_status.lower():
            backend_status = "limit_feasible"

        x = None
        if backend_status in {"optimal", "limit_feasible"}:
            sol = h.getSolution()
            candidate = np.asarray(list(sol.col_value), dtype=np.float64)
            if candidate.shape == (problem.n_variables,) and np.all(np.isfinite(candidate)):
                x = candidate

        objective = None
        objective_internal = None
        if x is not None:
            try:
                candidate = float(h.getObjectiveValue())
                if np.isfinite(candidate):
                    objective_internal = candidate
                    objective = candidate
            except Exception:
                objective_internal = None

        raw = {
            "model_status": h.modelStatusToString(model_status_enum),
            "primal_solution_status": primal_status,
            "simplex_iteration_count": int(getattr(info, "simplex_iteration_count", 0)),
            "ipm_iteration_count": int(getattr(info, "ipm_iteration_count", 0)),
            "qp_iteration_count": int(getattr(info, "qp_iteration_count", 0)),
            "mip_node_count": int(getattr(info, "mip_node_count", 0)),
            "reuse_applied": reuse_applied,
            "reuse_mode": reuse_mode,
            "same_structure": bool(same_structure),
            "highs_run_delta_s": max(0.0, runtime_after - runtime_before),
            "highs_runtime_cumulative_s": runtime_after,
            "private_scipy_api": True,
            "objective_internal": objective_internal,
        }
        self._last_problem = problem
        self._last_x = None if x is None else np.array(x, copy=True)
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective,
            raw_statistics=raw,
        )

    def compute_iis(self, problem: LinearProblem) -> HighsDevIISResult:
        """Development verification of native HiGHS IIS plumbing.

        This is intentionally a small typed surface.  Bound-status integers are kept
        solver-native because the SciPy-vendored HiGHS version is 1.8.0 and this
        verifier must not pretend those enum semantics are our stable public schema.
        """
        if problem.has_integer_variables:
            raise NotImplementedError("M3 vendored IIS verifier is limited to continuous LP")
        core, h = self._rebuild(problem)
        h.run()
        if h.getModelStatus() != core.HighsModelStatus.kInfeasible:
            raise ValueError("IIS requested for a model not classified infeasible by HiGHS")
        iis = core.HighsIis()
        status = h.getIis(iis)
        if not _ok(core, status):
            raise RuntimeError(f"HiGHS getIis failed: {status}")
        return HighsDevIISResult(
            valid=bool(iis.valid),
            row_indices=tuple(int(x) for x in iis.row_index),
            row_bound_statuses=tuple(int(x) for x in iis.row_bound),
            col_indices=tuple(int(x) for x in iis.col_index),
            col_bound_statuses=tuple(int(x) for x in iis.col_bound),
            strategy=int(iis.strategy),
        )

