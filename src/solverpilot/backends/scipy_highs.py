from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain

from .base import BackendSolveResult, BackendUnavailableError


@dataclass(slots=True)
class ScipyHighsBackend:
    """Development bridge to the HiGHS implementation bundled by SciPy.

    This is intentionally not the final native highspy adapter. It exists to exercise
    the backend protocol, status normalization, and end-to-end validation in environments
    where highspy itself is unavailable.
    """

    time_limit_s: float | None = None
    mip_rel_gap: float | None = None
    presolve: bool | None = None

    @property
    def manifest(self) -> BackendManifest:
        version = None
        if self.is_available():
            import scipy
            version = scipy.__version__
        return BackendManifest(
            name="scipy-highs-bridge",
            version=version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.UNSUPPORTED,
                Capability.MIP_START: SupportLevel.UNSUPPORTED,
                Capability.BASIS_START: SupportLevel.UNSUPPORTED,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNSUPPORTED,
                Capability.IIS: SupportLevel.UNSUPPORTED,
            },
            metadata={
                "development_bridge": True,
                "underlying_solver": "HiGHS via scipy.optimize.milp",
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("scipy.optimize") is not None

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("SciPy optimize is not available")
        if isinstance(problem, QuadraticProblem):
            raise ValueError("scipy-highs-bridge does not support QuadraticProblem")
        if not isinstance(problem, LinearProblem):
            raise TypeError(f"unsupported problem type: {type(problem)!r}")

        from scipy.optimize import Bounds, LinearConstraint, milp

        c = np.array(problem.c, copy=True)
        if problem.objective_sense is ObjectiveSense.MAXIMIZE:
            c = -c

        integrality = np.zeros(problem.n_variables, dtype=np.int32)
        integer_mask = np.isin(
            problem.domains,
            [VariableDomain.INTEGER.value, VariableDomain.BINARY.value],
        )
        integrality[integer_mask] = 1

        options: dict[str, object] = {"disp": False}
        if self.time_limit_s is not None:
            if self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            options["time_limit"] = float(self.time_limit_s)
        if self.mip_rel_gap is not None:
            if self.mip_rel_gap < 0:
                raise ValueError("mip_rel_gap must be non-negative")
            options["mip_rel_gap"] = float(self.mip_rel_gap)
        if self.presolve is not None:
            options["presolve"] = bool(self.presolve)

        result = milp(
            c=c,
            integrality=integrality,
            bounds=Bounds(problem.variable_lower, problem.variable_upper),
            constraints=LinearConstraint(
                problem.A,
                problem.constraint_lower,
                problem.constraint_upper,
            ),
            options=options,
        )

        status = self._status_name(int(result.status), result.x is not None)
        x = None if result.x is None else np.asarray(result.x, dtype=np.float64)
        objective = None
        if x is not None:
            objective = float(problem.c @ x + problem.objective_offset)

        raw = {
            "scipy_status": int(result.status),
            "message": str(result.message),
            "success": bool(result.success),
        }
        for attr in ("mip_node_count", "mip_dual_bound", "mip_gap"):
            if hasattr(result, attr):
                value = getattr(result, attr)
                if value is not None:
                    raw[attr] = float(value)

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
