from __future__ import annotations

from dataclasses import dataclass, field
import importlib.util
from importlib.metadata import PackageNotFoundError, version

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError


def _casadi_version() -> str | None:
    try:
        return version("casadi")
    except PackageNotFoundError:
        try:
            import casadi as ca
            return getattr(ca, "__version__", None)
        except Exception:
            return None


def _plugin_available(plugin: str) -> bool:
    if importlib.util.find_spec("casadi") is None:
        return False
    try:
        import casadi as ca
        token = f"Conic::{plugin}"
        return token in str(ca.CasadiMeta_plugins()).split(";")
    except Exception:
        return False


def _dm_from_scipy(matrix: sparse.spmatrix):
    import casadi as ca
    csc = sparse.csc_matrix(matrix, dtype=np.float64, copy=True)
    csc.sum_duplicates(); csc.eliminate_zeros(); csc.sort_indices()
    sp = ca.Sparsity(csc.shape[0], csc.shape[1], csc.indptr.tolist(), csc.indices.tolist())
    return ca.DM(sp, csc.data.tolist()), sp


def _status_from_stats(stats: dict[str, object], *, has_candidate: bool) -> str:
    raw = " ".join(
        str(stats.get(k, "")) for k in ("return_status", "secondary_return_status", "unified_return_status")
    ).lower()
    if "infeasible or unbounded" in raw:
        return "infeasible_or_unbounded"
    if "primal infeasible" in raw or "not feasible" in raw or ("infeasible" in raw and "dual infeasible" not in raw):
        return "infeasible"
    if "dual infeasible" in raw or "unbounded" in raw:
        return "unbounded"
    if any(tok in raw for tok in ("time limit", "maximum iterations", "max_iter", "iteration limit")):
        return "limit_feasible" if has_candidate else "limit_no_solution"
    if bool(stats.get("success")):
        return "optimal" if has_candidate else "solver_error"
    return "converged_candidate" if has_candidate else "solver_error"


@dataclass(slots=True)
class CasadiConicBackend:
    """Development bridge to native conic plugins bundled with CasADi.

    This is intentionally provenance-explicit: it verifies underlying solver engines in
    environments where their dedicated Python packages are unavailable, but it is not a
    substitute for the public highspy/osqp/PySCIPOpt adapters.
    """

    plugin: str
    time_limit_s: float | None = None
    enable_primal_warm_start: bool = True
    _solver_cache: object | None = field(default=None, init=False, repr=False)
    _structure_hash_cache: str | None = field(default=None, init=False, repr=False)
    _last_x_cache: np.ndarray | None = field(default=None, init=False, repr=False)

    def __post_init__(self) -> None:
        self.plugin = str(self.plugin).lower()
        if self.plugin not in {"highs", "osqp", "cbc"}:
            raise ValueError("supported CasADi conic plugins: highs, osqp, cbc")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")

    @property
    def manifest(self) -> BackendManifest:
        common = {
            Capability.PRIMAL_START: SupportLevel.UNKNOWN,
            Capability.DUAL_START: SupportLevel.UNKNOWN,
            Capability.BASIS_START: SupportLevel.UNKNOWN,
            Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.UNKNOWN,
            Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
            Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNKNOWN,
            Capability.CALLBACK_PROGRESS: SupportLevel.UNSUPPORTED,
            Capability.IIS: SupportLevel.UNSUPPORTED,
            Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
        }
        if self.plugin == "osqp":
            caps = {
                **common,
                Capability.LP: SupportLevel.EMULATED_SAFE,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.UNSUPPORTED,
                Capability.PRIMAL_START: SupportLevel.NATIVE,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.EMULATED_SAFE,
            }
            metadata = {"underlying_solver": "OSQP", "no_integer_support": True, "supports_wall_time_budget": False}
        elif self.plugin == "cbc":
            caps = {
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.UNSUPPORTED,
                Capability.MILP: SupportLevel.NATIVE,
                **common,
            }
            metadata = {"underlying_solver": "CBC", "no_qp_support": True, "supports_wall_time_budget": False}
        else:
            caps = {
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.NATIVE,
                Capability.MILP: SupportLevel.NATIVE,
                **common,
            }
            metadata = {"underlying_solver": "HiGHS", "supports_wall_time_budget": True}
        return BackendManifest(
            name=f"casadi-{self.plugin}-bridge",
            version=_casadi_version(),
            capabilities=caps,
            metadata={
                **metadata,
                "native_adapter": False,
                "development_bridge": True,
                "verification_only": True,
                "bridge_package": "casadi",
                "bridge_package_version": _casadi_version(),
                "verified_bridge_package_versions": ("3.7.2",),
                "cross_library_backend": self.plugin in {"osqp", "cbc"},
            },
        )

    def is_available(self) -> bool:
        return _plugin_available(self.plugin)

    def _opts(self, discrete: list[bool]) -> dict[str, object]:
        opts: dict[str, object] = {"error_on_fail": False}
        if self.plugin in {"highs", "cbc"}:
            opts["discrete"] = discrete
        if self.plugin == "highs":
            inner: dict[str, object] = {"output_flag": False}
            if self.time_limit_s is not None:
                inner["time_limit"] = float(self.time_limit_s)
            opts["highs"] = inner
        elif self.plugin == "osqp":
            if self.time_limit_s is not None:
                raise NotImplementedError("CasADi-OSQP time budget mapping is not verified")
            opts["osqp"] = {
                "verbose": False,
                "polish": True,
                "eps_abs": 1e-7,
                "eps_rel": 1e-7,
                "max_iter": 10000,
            }
        elif self.plugin == "cbc":
            inner = {"logLevel": 0}
            # CBC option names vary across Osi/Cbc layers. Do not claim time-limit
            # enforcement until a dedicated integration test proves the mapping.
            if self.time_limit_s is not None:
                raise NotImplementedError("CasADi-CBC time budget mapping is not verified")
            opts["cbc"] = inner
        return opts

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError(f"CasADi conic plugin is unavailable: {self.plugin}")
        import casadi as ca

        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
            if self.plugin == "cbc":
                raise ValueError("CBC bridge does not support QP")
            linear = problem.linear
            P = ((problem.P + problem.P.T) * 0.5).tocsc()
        elif isinstance(problem, LinearProblem):
            linear = problem
            P = sparse.csc_matrix((linear.n_variables, linear.n_variables), dtype=np.float64)
            if linear.has_integer_variables and self.plugin == "osqp":
                raise ValueError("OSQP bridge does not support integer variables")
        else:
            raise TypeError(f"unsupported problem type: {type(problem)!r}")

        if isinstance(problem, QuadraticProblem) and linear.objective_sense is ObjectiveSense.MAXIMIZE:
            raise ValueError("convex QP bridge supports minimization only")

        n = linear.n_variables
        A_dm, A_sp = _dm_from_scipy(linear.A)
        P_dm, P_sp = _dm_from_scipy(P)
        # CBC asserts that the Hessian sparsity itself has no nonzeros for LP/MILP.
        if self.plugin == "cbc" and not isinstance(problem, QuadraticProblem):
            P_sp = ca.Sparsity(n, n)
            P_dm = ca.DM.zeros(P_sp)

        discrete = [d != VariableDomain.CONTINUOUS.value for d in linear.domains]
        g = np.asarray(linear.c, dtype=np.float64).copy()
        sign = 1.0
        if linear.objective_sense is ObjectiveSense.MAXIMIZE:
            g *= -1.0
            sign = -1.0

        reused_primal_start = False
        if self.plugin == "osqp" and self.enable_primal_warm_start and self._solver_cache is not None and self._structure_hash_cache == problem.structural_hash:
            solver = self._solver_cache
        else:
            solver = ca.conic(
                "solverpilot_casadi_conic",
                self.plugin,
                {"h": P_sp, "a": A_sp},
                self._opts(discrete),
            )
            if self.plugin == "osqp" and self.enable_primal_warm_start:
                self._solver_cache = solver
                self._structure_hash_cache = problem.structural_hash
                self._last_x_cache = None
        kwargs = dict(
            h=P_dm,
            g=ca.DM(g),
            a=A_dm,
            lba=ca.DM(np.asarray(linear.constraint_lower, dtype=np.float64)),
            uba=ca.DM(np.asarray(linear.constraint_upper, dtype=np.float64)),
            lbx=ca.DM(np.asarray(linear.variable_lower, dtype=np.float64)),
            ubx=ca.DM(np.asarray(linear.variable_upper, dtype=np.float64)),
        )
        if self.plugin == "osqp" and self.enable_primal_warm_start and self._last_x_cache is not None and self._last_x_cache.shape == (n,):
            kwargs["x0"] = ca.DM(self._last_x_cache)
            reused_primal_start = True
        error = None
        result = None
        try:
            result = solver(**kwargs)
        except Exception as exc:
            error = f"{type(exc).__name__}: {exc}"

        try:
            stats = dict(solver.stats())
        except Exception:
            stats = {}

        x = None
        objective_reported = None
        if result is not None and "x" in result:
            arr = np.asarray(result["x"], dtype=np.float64).reshape(-1)
            raw_status = " ".join(str(stats.get(k, "")) for k in ("return_status", "secondary_return_status", "unified_return_status")).lower()
            terminal_bad = any(tok in raw_status for tok in ("infeasible", "not feasible", "unbounded"))
            if arr.shape == (n,) and np.all(np.isfinite(arr)) and not terminal_bad:
                x = arr
                if self.plugin == "osqp" and self.enable_primal_warm_start:
                    self._last_x_cache = arr.copy()
                try:
                    solver_cost = float(result["cost"])
                    objective_reported = sign * solver_cost + float(linear.objective_offset)
                except Exception:
                    objective_reported = None

        backend_status = _status_from_stats(stats, has_candidate=x is not None)
        stats.update(
            {
                "casadi_plugin": self.plugin,
                "casadi_version": _casadi_version(),
                "error": error,
                "reuse_applied": bool(reused_primal_start),
                "reuse_mode": "casadi_osqp_primal_warm_start" if reused_primal_start else "cold_casadi_bridge_solve",
            }
        )
        return BackendSolveResult(
            backend_status=backend_status,
            x=x,
            objective_reported=objective_reported,
            raw_statistics=stats,
        )


@dataclass(slots=True)
class CasadiOSQPBridgeBackend(CasadiConicBackend):
    plugin: str = field(default="osqp", init=False)


@dataclass(slots=True)
class CasadiHighsBridgeBackend(CasadiConicBackend):
    plugin: str = field(default="highs", init=False)


@dataclass(slots=True)
class CasadiCBCBridgeBackend(CasadiConicBackend):
    plugin: str = field(default="cbc", init=False)
