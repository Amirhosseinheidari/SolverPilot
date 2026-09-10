from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from importlib.metadata import PackageNotFoundError, version

import numpy as np

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


def _initial_point(lower: np.ndarray, upper: np.ndarray) -> np.ndarray:
    x = np.zeros_like(lower, dtype=np.float64)
    both = np.isfinite(lower) & np.isfinite(upper)
    x[both] = 0.5 * (lower[both] + upper[both])
    only_lower = np.isfinite(lower) & ~np.isfinite(upper)
    x[only_lower] = np.maximum(0.0, lower[only_lower])
    only_upper = ~np.isfinite(lower) & np.isfinite(upper)
    x[only_upper] = np.minimum(0.0, upper[only_upper])
    return np.minimum(np.maximum(x, lower), upper)


def _nlopt_status(code: int, *, has_candidate: bool) -> str:
    # NLopt positive result codes indicate a successful stop criterion, but not a
    # mathematical optimality certificate. Preserve that distinction.
    if code in {1, 2, 3, 4}:
        return "converged_candidate" if has_candidate else "solver_error"
    if code in {5, 6}:  # max evaluations / max time
        return "limit_feasible" if has_candidate else "limit_no_solution"
    if code in {-4, -5}:  # roundoff-limited / forced stop
        return "limit_feasible" if has_candidate else "limit_no_solution"
    return "solver_error"


@dataclass(slots=True)
class NLoptNativeBackend:
    """Optional cross-library continuous optimizer backend using NLopt SLSQP.

    This adapter is deliberately conservative: it accepts continuous LP and convex
    QP instances but never claims a proof of optimality. A positive NLopt stop is
    normalized as a candidate and SolverPilot's independent validator remains the
    authority on feasibility.
    """

    time_limit_s: float | None = None
    max_evals: int = 2000
    ftol_rel: float = 1e-10
    xtol_rel: float = 1e-10
    constraint_tol: float = 1e-8

    def __post_init__(self) -> None:
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.max_evals <= 0:
            raise ValueError("max_evals must be positive")
        if self.ftol_rel <= 0 or self.xtol_rel <= 0 or self.constraint_tol < 0:
            raise ValueError("NLopt tolerances must be positive (constraint_tol may be zero)")

    @property
    def manifest(self) -> BackendManifest:
        pkg_version = None
        if self.is_available():
            try:
                pkg_version = version("nlopt")
            except PackageNotFoundError:
                try:
                    import nlopt
                    pkg_version = getattr(nlopt, "__version__", None)
                except Exception:
                    pass
        return BackendManifest(
            name="nlopt-slsqp-native",
            version=pkg_version,
            capabilities={
                Capability.LP: SupportLevel.EMULATED_SAFE,
                Capability.CONVEX_QP: SupportLevel.EMULATED_SAFE,
                Capability.MILP: SupportLevel.UNSUPPORTED,
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
                "native_adapter": True,
                "underlying_solver": "NLopt LD_SLSQP",
                "continuous_only": True,
                "no_optimality_certificate": True,
                "cross_library_backend": True,
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("nlopt") is not None

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("nlopt is not installed")
        import nlopt

        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
            linear = problem.linear
            P = ((problem.P + problem.P.T) * 0.5).tocsr()
        elif isinstance(problem, LinearProblem):
            if problem.has_integer_variables:
                raise ValueError("NLopt backend supports continuous LP only")
            linear = problem
            P = None
        else:
            raise TypeError(f"unsupported problem type: {type(problem)!r}")

        n = linear.n_variables
        opt = nlopt.opt(nlopt.LD_SLSQP, n)
        opt.set_lower_bounds(np.asarray(linear.variable_lower, dtype=np.float64))
        opt.set_upper_bounds(np.asarray(linear.variable_upper, dtype=np.float64))
        opt.set_maxeval(int(self.max_evals))
        opt.set_ftol_rel(float(self.ftol_rel))
        opt.set_xtol_rel(float(self.xtol_rel))
        if self.time_limit_s is not None:
            opt.set_maxtime(float(self.time_limit_s))

        if P is None:
            c = np.asarray(linear.c, dtype=np.float64)

            def objective(x, grad):
                if grad.size:
                    grad[:] = c
                return float(c @ x + linear.objective_offset)
        else:
            q = np.asarray(linear.c, dtype=np.float64)

            def objective(x, grad):
                px = np.asarray(P @ x, dtype=np.float64)
                if grad.size:
                    grad[:] = px + q
                return float(0.5 * x @ px + q @ x + linear.objective_offset)

        if linear.objective_sense is ObjectiveSense.MAXIMIZE:
            if P is not None:
                raise ValueError("convex QuadraticProblem only supports minimization")
            opt.set_max_objective(objective)
        else:
            opt.set_min_objective(objective)

        A = linear.A.tocsr()
        tol = float(self.constraint_tol)
        callbacks = []  # keep closures alive for the lifetime of optimize()
        for i in range(linear.n_constraints):
            start, end = A.indptr[i], A.indptr[i + 1]
            indices = np.asarray(A.indices[start:end], dtype=np.int64).copy()
            values = np.asarray(A.data[start:end], dtype=np.float64).copy()
            lo = float(linear.constraint_lower[i])
            hi = float(linear.constraint_upper[i])

            def row_value(x, indices=indices, values=values):
                return float(values @ x[indices]) if values.size else 0.0

            if np.isfinite(lo) and np.isfinite(hi) and np.isclose(lo, hi, rtol=0.0, atol=0.0):
                rhs = hi

                def eq(x, grad, indices=indices, values=values, rhs=rhs):
                    if grad.size:
                        grad[:] = 0.0
                        grad[indices] = values
                    return float((values @ x[indices] if values.size else 0.0) - rhs)

                callbacks.append(eq)
                opt.add_equality_constraint(eq, tol)
                continue

            if np.isfinite(hi):
                rhs = hi

                def upper(x, grad, indices=indices, values=values, rhs=rhs):
                    if grad.size:
                        grad[:] = 0.0
                        grad[indices] = values
                    return float((values @ x[indices] if values.size else 0.0) - rhs)

                callbacks.append(upper)
                opt.add_inequality_constraint(upper, tol)

            if np.isfinite(lo):
                rhs = lo

                def lower(x, grad, indices=indices, values=values, rhs=rhs):
                    if grad.size:
                        grad[:] = 0.0
                        grad[indices] = -values
                    return float(rhs - (values @ x[indices] if values.size else 0.0))

                callbacks.append(lower)
                opt.add_inequality_constraint(lower, tol)

        x0 = _initial_point(linear.variable_lower, linear.variable_upper)
        x = None
        objective_reported = None
        code = None
        error = None
        try:
            candidate = np.asarray(opt.optimize(x0), dtype=np.float64)
            code = int(opt.last_optimize_result())
            if candidate.shape == (n,) and np.all(np.isfinite(candidate)):
                x = candidate
                objective_reported = float(opt.last_optimum_value())
        except Exception as exc:
            # NLopt exceptions may still leave a meaningful candidate unavailable to Python;
            # do not manufacture one. Preserve the exception in raw statistics.
            try:
                code = int(opt.last_optimize_result())
            except Exception:
                code = None
            error = f"{type(exc).__name__}: {exc}"

        backend_status = _nlopt_status(-1 if code is None else code, has_candidate=x is not None)
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective_reported,
            raw_statistics={
                "nlopt_result_code": code,
                "algorithm": "LD_SLSQP",
                "error": error,
                "reuse_applied": False,
                "reuse_mode": "cold_native_solve",
                "no_optimality_certificate": True,
            },
        )
