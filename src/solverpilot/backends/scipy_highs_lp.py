from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem

from .base import BackendSolveResult, BackendUnavailableError


@dataclass(slots=True)
class ScipyHighsLPBackend:
    """Development LP backend using ``scipy.optimize.linprog`` + HiGHS.

    Parameters
    ----------
    method:
        ``"highs-ds"`` for HiGHS dual simplex or ``"highs-ipm"`` for the
        HiGHS interior-point method.  This backend intentionally supports only
        continuous LPs; integer and quadratic models must be routed elsewhere.

    Notes
    -----
    This is a development bridge, not the eventual native ``highspy`` adapter.
    It exists so the adaptive runtime can exercise real algorithm alternatives
    without pretending SciPy exposes native HiGHS state reuse.
    """

    method: str = "highs-ds"
    time_limit_s: float | None = None
    presolve: bool | None = None

    def __post_init__(self) -> None:
        if self.method not in {"highs-ds", "highs-ipm"}:
            raise ValueError("method must be 'highs-ds' or 'highs-ipm'")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")

    @property
    def manifest(self) -> BackendManifest:
        version = None
        if self.is_available():
            import scipy

            version = scipy.__version__
        return BackendManifest(
            name=f"scipy-{self.method}",
            version=version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.UNSUPPORTED,
                Capability.CONVEX_QP: SupportLevel.UNSUPPORTED,
                Capability.MIP_START: SupportLevel.UNSUPPORTED,
                Capability.PRIMAL_START: SupportLevel.UNSUPPORTED,
                Capability.DUAL_START: SupportLevel.UNSUPPORTED,
                Capability.BASIS_START: SupportLevel.UNSUPPORTED,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNSUPPORTED,
                Capability.IIS: SupportLevel.UNSUPPORTED,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNSUPPORTED,
            },
            metadata={
                "development_bridge": True,
                "underlying_solver": f"HiGHS {self.method} via scipy.optimize.linprog",
                "algorithm": "dual_simplex" if self.method == "highs-ds" else "interior_point",
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("scipy.optimize") is not None

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("SciPy optimize is not available")
        if isinstance(problem, QuadraticProblem):
            raise ValueError(f"{self.manifest.name} does not support QuadraticProblem")
        if not isinstance(problem, LinearProblem):
            raise TypeError(f"unsupported problem type: {type(problem)!r}")
        if problem.has_integer_variables:
            raise ValueError(f"{self.manifest.name} supports continuous LPs only")

        from scipy.optimize import linprog

        c = np.array(problem.c, copy=True)
        if problem.objective_sense is ObjectiveSense.MAXIMIZE:
            c = -c

        A_ub, b_ub, A_eq, b_eq = _to_linprog_constraints(problem)
        bounds = list(zip(problem.variable_lower.tolist(), problem.variable_upper.tolist()))

        options: dict[str, object] = {"disp": False}
        if self.time_limit_s is not None:
            options["time_limit"] = float(self.time_limit_s)
        if self.presolve is not None:
            options["presolve"] = bool(self.presolve)

        result = linprog(
            c,
            A_ub=A_ub,
            b_ub=b_ub,
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds,
            method=self.method,
            options=options,
        )

        status = self._status_name(int(result.status), result.x is not None)
        x = None if result.x is None else np.asarray(result.x, dtype=np.float64)
        objective = None
        if x is not None:
            objective = float(problem.c @ x + problem.objective_offset)

        raw: dict[str, object] = {
            "scipy_status": int(result.status),
            "message": str(result.message),
            "success": bool(result.success),
            "nit": int(getattr(result, "nit", 0) or 0),
            "method": self.method,
        }
        if hasattr(result, "crossover_nit") and result.crossover_nit is not None:
            raw["crossover_nit"] = int(result.crossover_nit)

        return BackendSolveResult(
            backend_status=status,
            x=x,
            objective_reported=objective,
            raw_statistics=raw,
        )

    @staticmethod
    def _status_name(status: int, has_solution: bool) -> str:
        if status == 0:
            return "optimal"
        if status == 1:
            return "limit_feasible" if has_solution else "limit_no_solution"
        if status == 2:
            return "infeasible"
        if status == 3:
            return "unbounded"
        if status == 4:
            return "solver_error"
        return "unknown"


def _to_linprog_constraints(
    problem: LinearProblem,
) -> tuple[
    sparse.csr_matrix | None,
    np.ndarray | None,
    sparse.csr_matrix | None,
    np.ndarray | None,
]:
    """Translate ranged rows into linprog equality/upper-bound form.

    A row ``l <= a @ x <= u`` becomes:
    - equality if finite ``l == u``;
    - ``a @ x <= u`` when ``u`` is finite;
    - ``-a @ x <= -l`` when ``l`` is finite.

    The canonical problem remains the source of truth for final validation.
    """

    ub_rows: list[sparse.csr_matrix] = []
    ub_rhs: list[float] = []
    eq_rows: list[sparse.csr_matrix] = []
    eq_rhs: list[float] = []

    for i in range(problem.n_constraints):
        lo = float(problem.constraint_lower[i])
        hi = float(problem.constraint_upper[i])
        row = problem.A.getrow(i)

        if np.isfinite(lo) and np.isfinite(hi) and lo == hi:
            eq_rows.append(row)
            eq_rhs.append(lo)
            continue
        if np.isfinite(hi):
            ub_rows.append(row)
            ub_rhs.append(hi)
        if np.isfinite(lo):
            ub_rows.append(-row)
            ub_rhs.append(-lo)

    A_ub = sparse.vstack(ub_rows, format="csr") if ub_rows else None
    b_ub = np.asarray(ub_rhs, dtype=np.float64) if ub_rows else None
    A_eq = sparse.vstack(eq_rows, format="csr") if eq_rows else None
    b_eq = np.asarray(eq_rhs, dtype=np.float64) if eq_rows else None
    return A_ub, b_ub, A_eq, b_eq
