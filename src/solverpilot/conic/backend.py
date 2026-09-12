from __future__ import annotations

from dataclasses import dataclass
import importlib.util
from importlib.metadata import PackageNotFoundError, version
from typing import Any

import numpy as np

from .ir import ConeKind, ConicProblem
from .validation import ConicValidationReport, validate_conic_solution


def _casadi_version() -> str | None:
    try:
        return version("casadi")
    except PackageNotFoundError:
        try:
            import casadi as ca
            return getattr(ca, "__version__", None)
        except Exception:
            return None


def _superscs_available() -> bool:
    if importlib.util.find_spec("casadi") is None:
        return False
    try:
        import casadi as ca
        return bool(ca.has_conic("superscs"))
    except Exception:
        return False


@dataclass(frozen=True, slots=True)
class ConicSolveResult:
    backend: str
    backend_status: str
    x: np.ndarray | None
    objective_reported: float | None
    validation: ConicValidationReport
    raw_statistics: dict[str, Any]
    problem_data_hash: str | None = None

    def __post_init__(self):
        from solverpilot._immutability import deep_freeze, readonly_array
        if self.x is not None:
            object.__setattr__(self, 'x', readonly_array(self.x, dtype=float))
        object.__setattr__(self, 'raw_statistics', deep_freeze(self.raw_statistics))

    @property
    def validated(self) -> bool:
        return bool(self.validation.valid)

    @property
    def optimality_evidence(self):
        from solverpilot.runtime.result import OptimalityEvidence
        trust = self.raw_statistics.get("solverpilot_trust", {})
        return OptimalityEvidence(backend_reported_optimal=bool(self.raw_statistics.get("backend_reported_optimal")),
            primal_validated=self.validation.valid, dual_verified=bool(trust.get("dual_verified")),
            gap_verified=bool(trust.get("gap_verified")))


@dataclass(slots=True)
class CasadiSuperSCSBackend:
    """Verified P6 bridge to the SuperSCS plugin bundled by CasADi 3.7.2.

    This adapter is intentionally labeled a bridge, not a public SuperSCS Python
    binding. It is used because dedicated Clarabel/SCS packages are unavailable in
    the current release environment. Verified P6 solve scope is linear-objective SOC
    and rotated SOC (the latter via an exact linear map to SOC). PSD is represented
    and independently validated by P6 but intentionally solve-disabled for this adapter
    after negative objective-accuracy conformance.
    """

    max_iter: int = 20_000
    eps: float = 1e-8
    verbose: bool = False

    @property
    def name(self) -> str:
        return "casadi-superscs-conic-bridge"

    @property
    def capability_manifest_v2(self):
        from solverpilot import __version__
        from solverpilot.capabilities.v2 import (
            BackendCapabilityManifestV2, CapabilityClaim, CapabilityEvidence, CapabilityKey,
            CapabilityMode, CapabilityStatus, VerificationLevel,
        )
        binding_version = self.binding_version
        backend_version = f"bundled-with-casadi-{binding_version or 'unknown'}"
        evidence = CapabilityEvidence(
            evidence_id="p6-casadi-superscs-conformance-v1",
            kind="native-runtime-conformance",
            backend_versions=(backend_version,),
            binding_versions=(() if binding_version is None else (binding_version,)),
            adapter_versions=(__version__,),
            verified_on="2026-09-06",
            notes=(
                "CasADi SuperSCS plugin exercised on SOC/rotated-SOC models; PSD negative conformance retained",
                "rotated SOC is exact adapter transformation to SOC",
            ),
        )
        def verified(mode=CapabilityMode.NATIVE, restrictions=None):
            restrictions = {} if restrictions is None else restrictions
            status = CapabilityStatus.SUPPORTED if not restrictions else CapabilityStatus.RESTRICTED
            return CapabilityClaim(status=status, mode=mode, verification=VerificationLevel.VERIFIED, restrictions=restrictions, evidence=(evidence,))
        claims = {
            CapabilityKey.PROBLEM_CONIC: verified(CapabilityMode.EMULATED_SAFE, {"binding": "CasADi SuperSCS bridge", "continuous_only": True, "verified_objective": "linear"}),
            CapabilityKey.PROBLEM_CONIC_QUADRATIC: CapabilityClaim(status=CapabilityStatus.UNVERIFIED, mode=CapabilityMode.UNKNOWN, verification=VerificationLevel.UNVERIFIED, notes=("P6 generic quadratic-SOCP conformance failed on feasible analytic projection cases",)),
            CapabilityKey.CONSTRAINT_LINEAR: verified(CapabilityMode.NATIVE),
            CapabilityKey.CONSTRAINT_SOC: verified(CapabilityMode.EMULATED_SAFE, {"transport": "PSD Schur-complement representation via casadi.soc"}),
            CapabilityKey.CONSTRAINT_ROTATED_SOC: verified(CapabilityMode.EMULATED_SAFE, {"transport": "exact linear map to SOC then PSD representation"}),
            CapabilityKey.CONSTRAINT_PSD: CapabilityClaim(status=CapabilityStatus.UNVERIFIED, mode=CapabilityMode.UNKNOWN, verification=VerificationLevel.UNVERIFIED, notes=("P6 PSD semantic IR is implemented, but bundled SuperSCS failed objective-accuracy conformance for generic PSD programs",)),
            CapabilityKey.RESULT_PRIMAL: verified(CapabilityMode.NATIVE),
            CapabilityKey.RESULT_OBJECTIVE: verified(CapabilityMode.NATIVE),
        }
        return BackendCapabilityManifestV2(
            backend=self.name,
            backend_version=backend_version,
            binding="casadi",
            binding_version=binding_version,
            adapter_version=__version__,
            claims=claims,
            numeric_limits={"p6_tolerance": self.eps},
            metadata={"verification_only_bridge": True, "plugin": "superscs"},
        )

    @property
    def binding_version(self) -> str | None:
        return _casadi_version()

    def is_available(self) -> bool:
        return _superscs_available()

    def _build_solver(self, problem: ConicProblem):
        import casadi as ca

        n = problem.n_variables
        x = ca.MX.sym("x", n)
        P = ca.DM(problem.P.toarray())
        q = ca.DM(problem.q)
        f = 0.5 * ca.mtimes([x.T, P, x]) + ca.dot(q, x) + float(problem.objective_offset)

        g_exprs = []
        lbg: list[float] = []
        ubg: list[float] = []
        if problem.n_linear_constraints:
            Ad = ca.DM(problem.A.toarray())
            lin = ca.mtimes(Ad, x)
            g_exprs.append(lin)
            lbg.extend(problem.constraint_lower.tolist())
            ubg.extend(problem.constraint_upper.tolist())

        psd_blocks = []
        for block in problem.cones:
            F = ca.DM(block.F.toarray())
            g = ca.DM(block.g)
            y = ca.mtimes(F, x) + g
            if block.kind is ConeKind.SECOND_ORDER:
                psd_blocks.append(ca.soc(y[1:], y[0]))
            elif block.kind is ConeKind.ROTATED_SECOND_ORDER:
                # 2*u*v >= ||z||^2, u,v>=0  <=>
                # ||[sqrt(2)z, u-v]|| <= u+v.
                u = y[0]; v = y[1]; z = y[2:]
                soc_vec = ca.vertcat(np.sqrt(2.0) * z, u - v)
                psd_blocks.append(ca.soc(soc_vec, u + v))
                # Explicit nonnegativity keeps the semantic boundary obvious and
                # protects against numerical interpretations of the transformed SOC.
                g_exprs.append(ca.vertcat(u, v))
                lbg.extend([0.0, 0.0]); ubg.extend([np.inf, np.inf])
            elif block.kind is ConeKind.POSITIVE_SEMIDEFINITE:
                rows, cols = block.output_shape
                M = ca.reshape(y, cols, rows).T  # row-major semantic flattening
                psd_blocks.append(M)
            else:  # pragma: no cover
                raise ValueError(block.kind)

        h = None
        if psd_blocks:
            h = psd_blocks[0]
            for blk in psd_blocks[1:]:
                r1, c1 = h.shape; r2, c2 = blk.shape
                h = ca.blockcat([[h, ca.MX.zeros(r1, c2)], [ca.MX.zeros(r2, c1), blk]])

        qp: dict[str, Any] = {"x": x, "f": f}
        if g_exprs:
            qp["g"] = ca.vertcat(*g_exprs)
        if h is not None:
            qp["h"] = h

        opts: dict[str, Any] = {"error_on_fail": False}
        # SuperSCS option names follow its native interface. Keep the set minimal;
        # unsupported verbosity options have differed across CasADi builds.
        opts["superscs"] = {
            "max_iters": int(self.max_iter),
            "eps": float(self.eps),
            "verbose": 1 if self.verbose else 0,
        }
        try:
            solver = ca.qpsol("solverpilot_p6_superscs", "superscs", qp, opts)
        except Exception:
            # Some bundled SuperSCS builds reject the verbose key. Retry with only
            # convergence controls rather than silently abandoning the backend.
            opts["superscs"] = {"max_iters": int(self.max_iter), "eps": float(self.eps)}
            solver = ca.qpsol("solverpilot_p6_superscs", "superscs", qp, opts)
        return solver, np.asarray(lbg, dtype=float), np.asarray(ubg, dtype=float)

    def solve(self, problem: ConicProblem) -> ConicSolveResult:
        if not self.is_available():
            raise RuntimeError("CasADi SuperSCS conic plugin is unavailable")
        if problem.P.nnz:
            raise RuntimeError("P6 fail-closed: quadratic conic objective conformance did not pass for the bundled CasADi SuperSCS plugin")
        if any(block.kind is ConeKind.POSITIVE_SEMIDEFINITE for block in problem.cones):
            raise RuntimeError("P6 fail-closed: generic PSD solve conformance did not pass for the bundled CasADi SuperSCS plugin; install a future verified PSD backend instead")
        solver, lbg, ubg = self._build_solver(problem)
        kwargs: dict[str, Any] = {
            "lbx": problem.variable_lower,
            "ubx": problem.variable_upper,
        }
        if lbg.size:
            kwargs["lbg"] = lbg
            kwargs["ubg"] = ubg
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
        objective = None
        if result is not None and "x" in result:
            arr = np.asarray(result["x"], dtype=float).reshape(-1)
            if arr.shape == (problem.n_variables,) and np.all(np.isfinite(arr)):
                x = arr
                objective = problem.objective_value(arr)
        raw = " ".join(str(stats.get(k, "")) for k in ("return_status", "unified_return_status")).lower()
        if bool(stats.get("success")) and x is not None:
            status = "optimal_candidate"
        elif "infeasible" in raw:
            status = "infeasible"
        elif "unbounded" in raw:
            status = "unbounded"
        elif x is not None:
            status = "candidate"
        else:
            status = "solver_error"
        validation_tol = max(self.eps * 25.0, 5e-7)
        validation = validate_conic_solution(problem, x, atol=validation_tol, rtol=validation_tol)
        stats["independent_validation_tolerance"] = validation_tol
        stats.update({
            "casadi_version": _casadi_version(),
            "conic_plugin": "superscs",
            "error": error,
            "verification_only_bridge": True,
            "cone_families": sorted({c.kind.value for c in problem.cones}),
            "adapter_transformations": (["rotated_second_order->second_order_exact_linear_map"] if any(c.kind is ConeKind.ROTATED_SECOND_ORDER for c in problem.cones) else []),
        })
        return ConicSolveResult(self.name, status, x, objective, validation, stats, problem.data_hash)


def solve_conic(problem: ConicProblem, *, backend=None, budget=None, tolerances=None,
                progress=None, cancellation=None) -> ConicSolveResult:
    if backend is None:
        from .clarabel_backend import ClarabelBackend
        backend = ClarabelBackend() if ClarabelBackend().is_available() else CasadiSuperSCSBackend()
    if isinstance(backend, str):
        from solverpilot.runtime.catalog import backend_catalog
        backend = backend_catalog()[backend]
    from solverpilot.runtime.budgeting import configured_backend
    from solverpilot.exceptions import BudgetNotSupportedError
    updates = {}
    if budget is not None:
        if budget.memory_mb is not None:
            raise BudgetNotSupportedError('conic memory limits require process isolation')
        for field, value in [('time_limit_s', budget.wall_time_s), ('threads', budget.threads)]:
            if value is not None:
                if not hasattr(backend, field):
                    raise BudgetNotSupportedError(f'{backend.name} cannot enforce {field}')
                updates[field] = value
    backend = configured_backend(backend, updates)
    from solverpilot.capabilities.v2 import compatible_v2, requirements_v2_for
    manifest = backend.capability_manifest_v2
    ok, checks = compatible_v2(
        manifest, requirements_v2_for(problem),
        allow_safe_emulation=True, allow_risky_emulation=False, require_verified=True,
    )
    if not ok:
        reasons = "; ".join(f"{c.key.value}: {c.reason}" for c in checks if not c.usable)
        raise RuntimeError(f"P6 conic backend capability gate rejected solve: {reasons}")
    controls = {k: v for k, v in [('tolerances', tolerances), ('progress', progress), ('cancellation', cancellation)] if v is not None}
    if controls and backend.name != 'clarabel-native':
        raise BudgetNotSupportedError(f'{backend.name} does not support these common controls; use Clarabel')
    result = backend.solve(problem, **controls)
    from dataclasses import replace
    from solverpilot.runtime.manifest import json_value
    raw = dict(result.raw_statistics)
    raw['run_parameters'] = {**dict(raw.get('run_parameters', {})), 'budget': json_value(budget)}
    return replace(result, raw_statistics=raw)
