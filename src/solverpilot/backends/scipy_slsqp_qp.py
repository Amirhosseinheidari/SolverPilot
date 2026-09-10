from __future__ import annotations

from dataclasses import dataclass
import importlib.util

import numpy as np

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


@dataclass(slots=True)
class ScipySLSQPQPBackend:
    """Small-problem development bridge for convex QP via SciPy SLSQP.

    This backend exists only to exercise the full QP runtime path when OSQP is not
    installed. It deliberately advertises ``EMULATED_SAFE`` rather than native QP
    support and does not claim an optimality certificate. A successful SLSQP exit
    is normalized as a validated feasible candidate by the runtime.
    """

    maxiter: int = 1000
    ftol: float = 1e-10

    def __post_init__(self) -> None:
        if self.maxiter <= 0:
            raise ValueError("maxiter must be positive")
        if self.ftol <= 0:
            raise ValueError("ftol must be positive")

    @property
    def manifest(self) -> BackendManifest:
        version = None
        if self.is_available():
            import scipy

            version = scipy.__version__
        return BackendManifest(
            name="scipy-slsqp-qp-bridge",
            version=version,
            capabilities={
                Capability.LP: SupportLevel.UNSUPPORTED,
                Capability.MILP: SupportLevel.UNSUPPORTED,
                Capability.CONVEX_QP: SupportLevel.EMULATED_SAFE,
                Capability.PRIMAL_START: SupportLevel.UNSUPPORTED,
                Capability.DUAL_START: SupportLevel.UNSUPPORTED,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNSUPPORTED,
            },
            metadata={
                "development_bridge": True,
                "underlying_solver": "scipy.optimize.minimize(method='SLSQP')",
                "no_optimality_certificate": True,
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("scipy.optimize") is not None

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("SciPy optimize is not available")
        if not isinstance(problem, QuadraticProblem):
            raise ValueError("scipy-slsqp-qp-bridge supports QuadraticProblem only")

        from scipy.optimize import Bounds, LinearConstraint, minimize

        linear = problem.linear
        P = ((problem.P + problem.P.T) * 0.5).tocsr()
        q = linear.c

        def fun(x: np.ndarray) -> float:
            return float(0.5 * x @ (P @ x) + q @ x + linear.objective_offset)

        def jac(x: np.ndarray) -> np.ndarray:
            return np.asarray(P @ x + q, dtype=np.float64)

        x0 = _initial_point(linear.variable_lower, linear.variable_upper)
        constraints = []
        if linear.n_constraints:
            constraints = [
                LinearConstraint(
                    linear.A,
                    linear.constraint_lower,
                    linear.constraint_upper,
                )
            ]
        result = minimize(
            fun,
            x0,
            jac=jac,
            method="SLSQP",
            bounds=Bounds(linear.variable_lower, linear.variable_upper),
            constraints=constraints,
            options={
                "disp": False,
                "maxiter": int(self.maxiter),
                "ftol": float(self.ftol),
            },
        )

        x = None if result.x is None else np.asarray(result.x, dtype=np.float64)
        status = "converged_candidate" if bool(result.success) else (
            "limit_feasible" if x is not None else "solver_error"
        )
        objective = None if x is None else fun(x)
        return BackendSolveResult(
            backend_status=status,
            x=x,
            objective_reported=objective,
            raw_statistics={
                "scipy_status": int(result.status),
                "message": str(result.message),
                "success": bool(result.success),
                "nit": int(getattr(result, "nit", 0) or 0),
                "development_only": True,
            },
        )


def _initial_point(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x = np.zeros_like(lower, dtype=np.float64)
    both = np.isfinite(lower) & np.isfinite(upper)
    x[both] = 0.5 * (lower[both] + upper[both])
    only_lower = np.isfinite(lower) & ~np.isfinite(upper)
    x[only_lower] = np.maximum(0.0, lower[only_lower])
    only_upper = ~np.isfinite(lower) & np.isfinite(upper)
    x[only_upper] = np.minimum(0.0, upper[only_upper])
    return x
