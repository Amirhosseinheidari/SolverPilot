from __future__ import annotations

from threading import RLock
from time import perf_counter
from solverpilot._synchronization import serialized

from dataclasses import dataclass, field
import importlib.util
from importlib.metadata import PackageNotFoundError
from .metadata import version

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


@dataclass(frozen=True, slots=True)
class OSQPData:
    P: sparse.csc_matrix
    q: np.ndarray
    A: sparse.csc_matrix
    l: np.ndarray
    u: np.ndarray


def _osqp_data(problem: QuadraticProblem) -> OSQPData:
    """Translate canonical QP data into OSQP's l <= A x <= u form.

    Canonical variable bounds are appended as identity rows. P is reduced to its
    upper triangle because OSQP's Python interface consumes only that triangle.
    """

    if not isinstance(problem, QuadraticProblem):
        raise TypeError("OSQP native backend supports QuadraticProblem only")
    n = problem.n_variables
    P = sparse.triu(problem.P, format="csc")
    base_A = problem.linear.A.tocsc()
    identity = sparse.eye(n, format="csc")
    A = sparse.vstack([base_A, identity], format="csc")
    l = np.concatenate([problem.linear.constraint_lower, problem.linear.variable_lower])
    u = np.concatenate([problem.linear.constraint_upper, problem.linear.variable_upper])
    return OSQPData(
        P=P,
        q=np.asarray(problem.linear.c, dtype=np.float64),
        A=A,
        l=np.asarray(l, dtype=np.float64),
        u=np.asarray(u, dtype=np.float64),
    )


def _osqp_status(status: str, has_solution: bool) -> str:
    s = status.strip().lower()
    if s == "solved":
        return "optimal"
    if s == "solved inaccurate":
        return "converged_candidate"
    if s == "primal infeasible":
        return "infeasible"
    if s == "primal infeasible inaccurate":
        return "infeasible_candidate"
    if s == "dual infeasible":
        return "unbounded"
    if s == "dual infeasible inaccurate":
        return "unbounded_candidate"
    if any(token in s for token in ("maximum iterations", "run time limit", "time limit")):
        return "limit_feasible" if has_solution else "limit_no_solution"
    if "non convex" in s or "error" in s:
        return "solver_error"
    return "unknown"


def _primal_certificate_check(data: OSQPData, certificate: np.ndarray | None, *, eps: float) -> dict[str, object]:
    if certificate is None:
        return {"valid": False, "reason": "missing_certificate"}
    v = np.asarray(certificate, dtype=np.float64)
    if v.shape != (data.A.shape[0],) or not np.all(np.isfinite(v)):
        return {"valid": False, "reason": "invalid_shape_or_nonfinite"}
    atv = np.asarray(data.A.T @ v).reshape(-1)
    stationarity_inf = float(np.max(np.abs(atv), initial=0.0))
    pos = v > 0.0
    neg = v < 0.0
    if np.any(pos & np.isposinf(data.u)) or np.any(neg & np.isneginf(data.l)):
        separating_value = float("inf")
    else:
        separating_value = float(np.dot(data.u[pos], v[pos]) + np.dot(data.l[neg], v[neg]))
    from solverpilot.validate._certificate_arithmetic import matvec, box_min, dot
    n = data.A.shape[1]
    minimum = box_min(matvec(data.A.T, v), data.l[-n:], data.u[-n:])
    bound = np.where(v > 0, data.u, data.l)
    active = v != 0
    valid = (minimum is not None and np.isfinite(bound[active]).all()
             and minimum > dot(v[active], bound[active]))
    return {
        "valid": bool(valid),
        "stationarity_inf": stationarity_inf,
        "separating_value": separating_value,
        "eps": float(eps),
    }


def _dual_certificate_check(problem: QuadraticProblem, data: OSQPData, certificate: np.ndarray | None, *, eps: float) -> dict[str, object]:
    if certificate is None:
        return {"valid": False, "reason": "missing_certificate"}
    s = np.asarray(certificate, dtype=np.float64)
    if s.shape != (problem.n_variables,) or not np.all(np.isfinite(s)):
        return {"valid": False, "reason": "invalid_shape_or_nonfinite"}
    ps = np.asarray(problem.P @ s).reshape(-1)
    ps_inf = float(np.max(np.abs(ps), initial=0.0))
    q_dot_s = float(problem.linear.c @ s)
    a_s = np.asarray(data.A @ s).reshape(-1)
    finite_l = np.isfinite(data.l)
    finite_u = np.isfinite(data.u)
    both = finite_l & finite_u
    lower_only = finite_l & ~finite_u
    upper_only = ~finite_l & finite_u
    row_violation = 0.0
    if np.any(both):
        row_violation = max(row_violation, float(np.max(np.abs(a_s[both]), initial=0.0)))
    if np.any(lower_only):
        row_violation = max(row_violation, float(np.max(np.maximum(-a_s[lower_only], 0.0), initial=0.0)))
    if np.any(upper_only):
        row_violation = max(row_violation, float(np.max(np.maximum(a_s[upper_only], 0.0), initial=0.0)))
    from solverpilot.validate._certificate_arithmetic import matvec, dot
    exact_rows = matvec(data.A, s)
    valid = (not any(matvec(problem.P, s)) and dot(problem.linear.c, s) < 0
             and all((not lo or value >= 0) and (not hi or value <= 0)
                     for value, lo, hi in zip(exact_rows, finite_l, finite_u)))
    return {
        "valid": bool(valid),
        "ps_inf": ps_inf,
        "q_dot_s": q_dot_s,
        "row_sign_violation": row_violation,
        "eps": float(eps),
    }


@dataclass(slots=True)
class OSQPNativeBackend:
    """Optional native OSQP adapter with same-sparsity state reuse.

    The adapter is import-safe when OSQP is absent. When the same backend object is
    reused for a QP whose structural hash is unchanged, it calls OSQP's documented
    ``update`` API and relies on OSQP's automatic primal/dual warm start.
    """

    time_limit_s: float | None = None
    max_iter: int | None = None
    eps_abs: float | None = None
    eps_rel: float | None = None
    eps_prim_inf: float = 1e-4
    eps_dual_inf: float = 1e-4
    polishing: bool = False
    _lock: object = field(default_factory=RLock, init=False, repr=False, compare=False)
    _solver: object | None = field(default=None, init=False, repr=False)
    _structural_hash: str | None = field(default=None, init=False, repr=False)

    @property
    def manifest(self) -> BackendManifest:
        pkg_version = None
        if self.is_available():
            try:
                pkg_version = version("osqp")
            except PackageNotFoundError:
                pass
        return BackendManifest(
            name="osqp-native",
            version=pkg_version,
            capabilities={
                Capability.LP: SupportLevel.UNSUPPORTED,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.UNSUPPORTED,
                Capability.MIP_START: SupportLevel.UNSUPPORTED,
                Capability.PRIMAL_START: SupportLevel.NATIVE,
                Capability.DUAL_START: SupportLevel.NATIVE,
                Capability.BASIS_START: SupportLevel.UNSUPPORTED,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNSUPPORTED,
                Capability.IIS: SupportLevel.UNSUPPORTED,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.NATIVE,
            },
            metadata={
                "native_adapter": True,
                "stateful": True,
                "reuse_semantics": "same sparsity update + OSQP automatic warm start",
            },
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("osqp") is not None

    def _settings(self) -> dict[str, object]:
        import osqp
        defaults = osqp.ext_builtin.OSQPSettings()
        osqp.ext_builtin.osqp_set_default_settings(defaults)
        settings: dict[str, object] = {
            "verbose": False,
            "warm_starting": True,
            "polishing": bool(self.polishing),
            "time_limit": defaults.time_limit,
            "max_iter": defaults.max_iter,
            "eps_abs": defaults.eps_abs,
            "eps_rel": defaults.eps_rel,
        }
        if self.time_limit_s is not None:
            if not np.isfinite(self.time_limit_s) or self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            settings["time_limit"] = float(self.time_limit_s)
        if self.max_iter is not None:
            if isinstance(self.max_iter, bool) or not isinstance(self.max_iter, (int, np.integer)) or self.max_iter <= 0:
                raise ValueError("max_iter must be positive")
            settings["max_iter"] = int(self.max_iter)
        if self.eps_abs is not None:
            if not np.isfinite(self.eps_abs) or self.eps_abs <= 0:
                raise ValueError("eps_abs must be positive")
            settings["eps_abs"] = float(self.eps_abs)
        if self.eps_rel is not None:
            if not np.isfinite(self.eps_rel) or self.eps_rel <= 0:
                raise ValueError("eps_rel must be positive")
            settings["eps_rel"] = float(self.eps_rel)
        if not np.isfinite([self.eps_prim_inf, self.eps_dual_inf]).all() or self.eps_prim_inf <= 0 or self.eps_dual_inf <= 0:
            raise ValueError("OSQP infeasibility tolerances must be positive")
        settings["eps_prim_inf"] = float(self.eps_prim_inf)
        settings["eps_dual_inf"] = float(self.eps_dual_inf)
        return settings

    @serialized
    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("OSQP is not installed")
        if not isinstance(problem, QuadraticProblem):
            raise ValueError("osqp-native supports convex continuous QP only")

        import osqp

        prepare_start = perf_counter()
        data = _osqp_data(problem)
        reuse_applied = self._solver is not None and self._structural_hash == problem.structural_hash
        if not reuse_applied:
            solver = osqp.OSQP()
            solver.setup(P=data.P, q=data.q, A=data.A, l=data.l, u=data.u, **self._settings())
            self._solver = solver
            self._structural_hash = problem.structural_hash
            reuse_mode = "cold_setup"
        else:
            # OSQP documents vector updates and P/A value updates with unchanged sparsity.
            # Supplying all numerical values avoids guessing which mutation occurred.
            self._solver.update(q=data.q, l=data.l, u=data.u, Px=data.P.data, Ax=data.A.data)
            # Mutable settings may be changed without setup; keep configured values aligned.
            update_settings = {
                k: v
                for k, v in self._settings().items()
                if k in {"verbose", "warm_starting", "polishing", "time_limit", "max_iter", "eps_abs", "eps_rel", "eps_prim_inf", "eps_dual_inf"}
            }
            if update_settings:
                self._solver.update_settings(**update_settings)
            reuse_mode = "same_sparsity_update+automatic_warm_start"

        prepared = perf_counter()
        result = self._solver.solve(raise_error=False)
        solved = perf_counter()
        info = result.info
        status_text = str(info.status)
        x_raw = getattr(result, "x", None)
        x_candidate = None if x_raw is None else np.asarray(x_raw, dtype=np.float64)
        has_finite_solution = x_candidate is not None and x_candidate.shape == (problem.n_variables,) and np.all(np.isfinite(x_candidate))
        backend_status = _osqp_status(status_text, has_finite_solution)
        certificate_kind = None
        certificate_check = None
        status_lower = status_text.strip().lower()
        if status_lower.startswith("primal infeasible"):
            certificate_kind = "primal_infeasibility"
            multiplier = 10.0 if status_lower.endswith("inaccurate") else 1.0
            certificate_check = _primal_certificate_check(
                data, getattr(result, "prim_inf_cert", None), eps=self.eps_prim_inf * multiplier
            )
            if status_lower == "primal infeasible" and not certificate_check.get("valid"):
                backend_status = "infeasible_candidate"
        elif status_lower.startswith("dual infeasible"):
            certificate_kind = "dual_infeasibility"
            multiplier = 10.0 if status_lower.endswith("inaccurate") else 1.0
            certificate_check = _dual_certificate_check(
                problem, data, getattr(result, "dual_inf_cert", None), eps=self.eps_dual_inf * multiplier
            )
            if status_lower == "dual infeasible" and not certificate_check.get("valid"):
                backend_status = "unbounded_candidate"

        x = x_candidate if backend_status in {"optimal", "converged_candidate", "limit_feasible"} and has_finite_solution else None

        objective = None
        objective_internal = None
        if x is not None:
            raw_obj = getattr(info, "obj_val", None)
            if raw_obj is not None and np.isfinite(float(raw_obj)):
                objective_internal = float(raw_obj)
                objective = objective_internal + float(problem.linear.objective_offset)

        raw: dict[str, object] = {
            "phase_timings": {"backend_build_s": 0. if reuse_applied else prepared-prepare_start, "backend_update_s": prepared-prepare_start if reuse_applied else 0., "solve_s": solved-prepared},
            "canonical_dual": None if x is None else np.asarray(result.y, dtype=float).tolist(),
            "osqp_status": status_text,
            "status_val": int(getattr(info, "status_val", 0)),
            "iter": int(getattr(info, "iter", 0)),
            "setup_time": float(getattr(info, "setup_time", 0.0)),
            "solve_time": float(getattr(info, "solve_time", 0.0)),
            "update_time": float(getattr(info, "update_time", 0.0)),
            "polish_time": float(getattr(info, "polish_time", 0.0)),
            "run_time": float(getattr(info, "run_time", 0.0)),
            "reuse_applied": bool(reuse_applied),
            "reuse_mode": reuse_mode,
            "certificate_kind": certificate_kind,
            "certificate_check": certificate_check,
            "objective_internal": objective_internal,
        }
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective,
            raw_statistics=raw,
        )
