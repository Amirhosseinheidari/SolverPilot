from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


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

    This adapter is import-safe in environments without ``highspy``. M2 implements
    native LP/MILP/QP model construction and result normalization. Stateful basis and
    MIP-start reuse are intentionally not claimed until they can be integration-tested.
    """

    time_limit_s: float | None = None
    threads: int | None = None
    presolve: bool | None = None
    solver: str | None = None

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
                # Upstream supports these mechanisms, but M2 adapter does not yet apply them.
                Capability.MIP_START: SupportLevel.UNKNOWN,
                Capability.PRIMAL_START: SupportLevel.UNKNOWN,
                Capability.DUAL_START: SupportLevel.UNKNOWN,
                Capability.BASIS_START: SupportLevel.UNKNOWN,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNKNOWN,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNKNOWN,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNKNOWN,
                Capability.IIS: SupportLevel.UNKNOWN,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
            },
            metadata={
                "native_adapter": True,
                "stateful": False,
                "reuse_claim": "none in M2 adapter",
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

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("highspy is not installed")
        import highspy

        h = highspy.Highs()
        h.setOptionValue("output_flag", False)
        if self.time_limit_s is not None:
            if self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            h.setOptionValue("time_limit", float(self.time_limit_s))
        if self.threads is not None:
            if self.threads <= 0:
                raise ValueError("threads must be positive")
            h.setOptionValue("threads", int(self.threads))
        if self.presolve is not None:
            h.setOptionValue("presolve", "on" if self.presolve else "off")
        if self.solver is not None:
            h.setOptionValue("solver", str(self.solver))

        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
            model = self._make_qp_model(highspy, problem)
        elif isinstance(problem, LinearProblem):
            model = self._make_lp_model(highspy, problem)
        else:
            raise TypeError(f"unsupported problem type: {type(problem)!r}")

        pass_status = h.passModel(model)
        if "error" in str(pass_status).lower():
            raise RuntimeError(f"HiGHS rejected model: {pass_status}")
        h.run()
        info = h.getInfo()
        model_status = h.modelStatusToString(h.getModelStatus())
        primal_status = h.solutionStatusToString(info.primal_solution_status)
        backend_status = _highs_status_from_strings(model_status, primal_status)

        x = None
        if backend_status in {"optimal", "limit_feasible"}:
            solution = h.getSolution()
            x_candidate = np.asarray(list(solution.col_value), dtype=np.float64)
            if x_candidate.shape == (problem.n_variables,) and np.all(np.isfinite(x_candidate)):
                x = x_candidate

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
            "model_status": str(model_status),
            "primal_solution_status": str(primal_status),
            "simplex_iteration_count": int(getattr(info, "simplex_iteration_count", 0)),
            "ipm_iteration_count": int(getattr(info, "ipm_iteration_count", 0)),
            "qp_iteration_count": int(getattr(info, "qp_iteration_count", 0)),
            "mip_node_count": int(getattr(info, "mip_node_count", 0)),
            "reuse_applied": False,
            "reuse_mode": "cold_native_solve",
            "objective_internal": objective_internal,
        }
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
