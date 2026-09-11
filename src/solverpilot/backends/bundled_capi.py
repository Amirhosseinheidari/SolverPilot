from __future__ import annotations

"""Direct C-API verification bridges for solver libraries bundled by CasADi.

These backends intentionally do *not* pretend to be the dedicated public Python
packages.  They exist to exercise the underlying solver C APIs directly in the
network-restricted verification environment used by this project.  Availability
is deliberately strict: only the CasADi build/version and solver ABI that were
verified by M13 are accepted.
"""

from dataclasses import dataclass, field
import ctypes
import hashlib
import importlib.util
from importlib.metadata import PackageNotFoundError
from .metadata import version
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse

from solverpilot.capabilities import BackendManifest, Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain
from solverpilot.problem.quadratic import require_confirmed_convexity

from .base import BackendSolveResult, BackendUnavailableError
from .osqp_native import _dual_certificate_check, _osqp_data, _osqp_status, _primal_certificate_check


_VERIFIED_CASADI_VERSION = "3.7.2"
_VERIFIED_OSQP_VERSION = "0.6.3"
_VERIFIED_HIGHS_VERSION = "1.10.0"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def _casadi_version() -> str | None:
    try:
        return version("casadi")
    except PackageNotFoundError:
        return None


def _casadi_dir() -> Path | None:
    spec = importlib.util.find_spec("casadi")
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).resolve().parent


def _bundled_paths() -> dict[str, Path] | None:
    root = _casadi_dir()
    if root is None:
        return None
    return {
        "root": root,
        "osqp_lib": root / "libosqp.so",
        "osqp_header": root / "include" / "osqp" / "osqp.h",
        "osqp_types": root / "include" / "osqp" / "types.h",
        "osqp_config": root / "include" / "osqp" / "osqp_configure.h",
        "highs_lib": root / "libhighs.so",
        "highs_header": root / "include" / "highs" / "interfaces" / "highs_c_api.h",
    }


# ---- OSQP 0.6.3 ABI verified against the headers bundled by CasADi 3.7.2 ----
_CInt = ctypes.c_longlong  # DLONG
_CFloat = ctypes.c_double  # !DFLOAT


class _CSC(ctypes.Structure):
    _fields_ = [
        ("nzmax", _CInt),
        ("m", _CInt),
        ("n", _CInt),
        ("p", ctypes.POINTER(_CInt)),
        ("i", ctypes.POINTER(_CInt)),
        ("x", ctypes.POINTER(_CFloat)),
        ("nz", _CInt),
    ]


class _OSQPData(ctypes.Structure):
    _fields_ = [
        ("n", _CInt),
        ("m", _CInt),
        ("P", ctypes.POINTER(_CSC)),
        ("A", ctypes.POINTER(_CSC)),
        ("q", ctypes.POINTER(_CFloat)),
        ("l", ctypes.POINTER(_CFloat)),
        ("u", ctypes.POINTER(_CFloat)),
    ]


class _OSQPSettings(ctypes.Structure):
    _fields_ = [
        ("rho", _CFloat),
        ("sigma", _CFloat),
        ("scaling", _CInt),
        ("adaptive_rho", _CInt),
        ("adaptive_rho_interval", _CInt),
        ("adaptive_rho_tolerance", _CFloat),
        ("max_iter", _CInt),
        ("eps_abs", _CFloat),
        ("eps_rel", _CFloat),
        ("eps_prim_inf", _CFloat),
        ("eps_dual_inf", _CFloat),
        ("alpha", _CFloat),
        ("linsys_solver", ctypes.c_int),
        ("delta", _CFloat),
        ("polish", _CInt),
        ("polish_refine_iter", _CInt),
        ("verbose", _CInt),
        ("scaled_termination", _CInt),
        ("check_termination", _CInt),
        ("warm_start", _CInt),
    ]


class _OSQPSolution(ctypes.Structure):
    _fields_ = [("x", ctypes.POINTER(_CFloat)), ("y", ctypes.POINTER(_CFloat))]


class _OSQPInfo(ctypes.Structure):
    _fields_ = [
        ("iter", _CInt),
        ("status", ctypes.c_char * 32),
        ("status_val", _CInt),
        ("status_polish", _CInt),
        ("obj_val", _CFloat),
        ("pri_res", _CFloat),
        ("dua_res", _CFloat),
        ("rho_updates", _CInt),
        ("rho_estimate", _CFloat),
    ]


class _OSQPWorkspace(ctypes.Structure):
    # PROFILING is disabled in the verified bundled OSQP build.  Only fields up
    # through `info` are needed, but preceding fields must preserve ABI offsets.
    _fields_ = [
        ("data", ctypes.POINTER(_OSQPData)),
        ("linsys_solver", ctypes.c_void_p),
        ("pol", ctypes.c_void_p),
        ("rho_vec", ctypes.POINTER(_CFloat)),
        ("rho_inv_vec", ctypes.POINTER(_CFloat)),
        ("constr_type", ctypes.POINTER(_CInt)),
        ("x", ctypes.POINTER(_CFloat)),
        ("y", ctypes.POINTER(_CFloat)),
        ("z", ctypes.POINTER(_CFloat)),
        ("xz_tilde", ctypes.POINTER(_CFloat)),
        ("x_prev", ctypes.POINTER(_CFloat)),
        ("z_prev", ctypes.POINTER(_CFloat)),
        ("Ax", ctypes.POINTER(_CFloat)),
        ("Px", ctypes.POINTER(_CFloat)),
        ("Aty", ctypes.POINTER(_CFloat)),
        ("delta_y", ctypes.POINTER(_CFloat)),
        ("Atdelta_y", ctypes.POINTER(_CFloat)),
        ("delta_x", ctypes.POINTER(_CFloat)),
        ("Pdelta_x", ctypes.POINTER(_CFloat)),
        ("Adelta_x", ctypes.POINTER(_CFloat)),
        ("D_temp", ctypes.POINTER(_CFloat)),
        ("D_temp_A", ctypes.POINTER(_CFloat)),
        ("E_temp", ctypes.POINTER(_CFloat)),
        ("settings", ctypes.POINTER(_OSQPSettings)),
        ("scaling", ctypes.c_void_p),
        ("solution", ctypes.POINTER(_OSQPSolution)),
        ("info", ctypes.POINTER(_OSQPInfo)),
        ("summary_printed", _CInt),
    ]


_OSQP_ABI_EXPECTED = {
    "sizeof_c_int": 8,
    "sizeof_c_float": 8,
    "sizeof_csc": 56,
    "sizeof_OSQPData": 56,
    "sizeof_OSQPSettings": 160,
    "sizeof_OSQPSolution": 16,
    "sizeof_OSQPInfo": 96,
    "sizeof_OSQPWorkspace": 224,
    "offset_ws_solution": 200,
    "offset_ws_info": 208,
    "offset_info_status": 8,
    "offset_info_obj": 56,
}


def _osqp_ctypes_layout() -> dict[str, int]:
    return {
        "sizeof_c_int": ctypes.sizeof(_CInt),
        "sizeof_c_float": ctypes.sizeof(_CFloat),
        "sizeof_csc": ctypes.sizeof(_CSC),
        "sizeof_OSQPData": ctypes.sizeof(_OSQPData),
        "sizeof_OSQPSettings": ctypes.sizeof(_OSQPSettings),
        "sizeof_OSQPSolution": ctypes.sizeof(_OSQPSolution),
        "sizeof_OSQPInfo": ctypes.sizeof(_OSQPInfo),
        "sizeof_OSQPWorkspace": ctypes.sizeof(_OSQPWorkspace),
        "offset_ws_solution": _OSQPWorkspace.solution.offset,
        "offset_ws_info": _OSQPWorkspace.info.offset,
        "offset_info_status": _OSQPInfo.status.offset,
        "offset_info_obj": _OSQPInfo.obj_val.offset,
    }


def _verified_osqp_config(paths: dict[str, Path]) -> bool:
    try:
        config = paths["osqp_config"].read_text(encoding="utf-8")
    except OSError:
        return False
    return (
        "#define DLONG" in config
        and "/* #undef DFLOAT */" in config
        and "/* #undef PROFILING */" in config
        and "/* #undef EMBEDDED */" in config
        and _osqp_ctypes_layout() == _OSQP_ABI_EXPECTED
    )


def _load_osqp_capi(paths: dict[str, Path]):
    lib = ctypes.CDLL(str(paths["osqp_lib"]))
    lib.osqp_version.restype = ctypes.c_char_p
    lib.osqp_set_default_settings.argtypes = [ctypes.POINTER(_OSQPSettings)]
    lib.osqp_setup.argtypes = [
        ctypes.POINTER(ctypes.POINTER(_OSQPWorkspace)),
        ctypes.POINTER(_OSQPData),
        ctypes.POINTER(_OSQPSettings),
    ]
    lib.osqp_setup.restype = _CInt
    lib.osqp_solve.argtypes = [ctypes.POINTER(_OSQPWorkspace)]
    lib.osqp_solve.restype = _CInt
    lib.osqp_cleanup.argtypes = [ctypes.POINTER(_OSQPWorkspace)]
    lib.osqp_cleanup.restype = _CInt
    lib.osqp_update_lin_cost.argtypes = [ctypes.POINTER(_OSQPWorkspace), ctypes.POINTER(_CFloat)]
    lib.osqp_update_lin_cost.restype = _CInt
    lib.osqp_update_bounds.argtypes = [
        ctypes.POINTER(_OSQPWorkspace),
        ctypes.POINTER(_CFloat),
        ctypes.POINTER(_CFloat),
    ]
    lib.osqp_update_bounds.restype = _CInt
    lib.osqp_update_P_A.argtypes = [
        ctypes.POINTER(_OSQPWorkspace),
        ctypes.POINTER(_CFloat),
        ctypes.POINTER(_CInt),
        _CInt,
        ctypes.POINTER(_CFloat),
        ctypes.POINTER(_CInt),
        _CInt,
    ]
    lib.osqp_update_P_A.restype = _CInt
    return lib


def _status_text(info: _OSQPInfo) -> str:
    return bytes(info.status).split(b"\0", 1)[0].decode("ascii", errors="replace")


def _c_int_array(values: np.ndarray):
    arr = np.asarray(values, dtype=np.int64)
    return (_CInt * len(arr))(*arr.tolist())


def _c_float_array(values: np.ndarray):
    arr = np.asarray(values, dtype=np.float64)
    return (_CFloat * len(arr))(*arr.tolist())


@dataclass(slots=True)
class _OSQPState:
    work: ctypes.POINTER(_OSQPWorkspace)
    keepalive: tuple[object, ...]
    structural_hash: str
    p_values: np.ndarray
    a_values: np.ndarray


@dataclass(slots=True)
class BundledOSQPCAPIBackend:
    """Direct OSQP C-API verification backend for CasADi's bundled library.

    Reuse evidence is stronger than the CasADi bridge: the OSQP workspace is
    preserved and the documented update functions are called directly.  We do
    *not* claim factorization reuse because matrix-value updates may refactorize.
    """

    max_iter: int = 10000
    eps_abs: float = 1e-7
    eps_rel: float = 1e-7
    eps_prim_inf: float = 1e-4
    eps_dual_inf: float = 1e-4
    polish: bool = True
    _state: _OSQPState | None = field(default=None, init=False, repr=False)

    @property
    def manifest(self) -> BackendManifest:
        paths = _bundled_paths()
        metadata: dict[str, object] = {
            "native_adapter": False,
            "verification_only": True,
            "direct_c_api": True,
            "public_python_package": False,
            "bundled_by": "casadi",
            "bridge_package_version": _casadi_version(),
            "verified_bridge_package_versions": (_VERIFIED_CASADI_VERSION,),
            "expected_solver_version": _VERIFIED_OSQP_VERSION,
            "verified_solver_versions": (_VERIFIED_OSQP_VERSION,),
            "supports_wall_time_budget": False,
            "stateful": True,
            "reuse_semantics": "persistent OSQP workspace + direct same-sparsity update API + warm_start=1",
            "factorization_reuse_claimed": False,
            "verified_abi_layout": dict(_OSQP_ABI_EXPECTED),
        }
        solver_version = None
        if paths and paths["osqp_lib"].is_file():
            try:
                lib = _load_osqp_capi(paths)
                solver_version = (lib.osqp_version() or b"").decode("ascii", errors="replace") or None
                metadata.update(
                    {
                        "library_sha256": _sha256(paths["osqp_lib"]),
                        "header_sha256": _sha256(paths["osqp_header"]),
                        "types_header_sha256": _sha256(paths["osqp_types"]),
                        "configure_header_sha256": _sha256(paths["osqp_config"]),
                    }
                )
            except Exception:
                pass
        return BackendManifest(
            name="bundled-osqp-capi",
            version=solver_version,
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
            metadata=metadata,
        )

    def is_available(self) -> bool:
        if _casadi_version() != _VERIFIED_CASADI_VERSION:
            return False
        paths = _bundled_paths()
        if paths is None or not all(paths[k].is_file() for k in ("osqp_lib", "osqp_header", "osqp_types", "osqp_config")):
            return False
        if not _verified_osqp_config(paths):
            return False
        try:
            lib = _load_osqp_capi(paths)
            return (lib.osqp_version() or b"").decode("ascii") == _VERIFIED_OSQP_VERSION
        except Exception:
            return False

    def close(self) -> None:
        if self._state is None:
            return
        try:
            paths = _bundled_paths()
            if paths is not None:
                _load_osqp_capi(paths).osqp_cleanup(self._state.work)
        finally:
            self._state = None

    def __del__(self):  # pragma: no cover - defensive process-exit cleanup
        try:
            self.close()
        except Exception:
            pass

    def _setup(self, problem: QuadraticProblem, data) -> tuple[object, float]:
        self.close()
        paths = _bundled_paths()
        if paths is None:
            raise RuntimeError("bundled OSQP C API paths became unavailable during setup")
        lib = _load_osqp_capi(paths)

        P = sparse.csc_matrix(data.P, dtype=np.float64, copy=True)
        A = sparse.csc_matrix(data.A, dtype=np.float64, copy=True)
        P.sort_indices(); A.sort_indices()
        Pp, Pi, Px = _c_int_array(P.indptr), _c_int_array(P.indices), _c_float_array(P.data)
        Ap, Ai, Ax = _c_int_array(A.indptr), _c_int_array(A.indices), _c_float_array(A.data)
        q, l, u = _c_float_array(data.q), _c_float_array(data.l), _c_float_array(data.u)
        P_csc = _CSC(P.nnz, P.shape[0], P.shape[1], Pp, Pi, Px, -1)
        A_csc = _CSC(A.nnz, A.shape[0], A.shape[1], Ap, Ai, Ax, -1)
        cdata = _OSQPData(problem.n_variables, A.shape[0], ctypes.pointer(P_csc), ctypes.pointer(A_csc), q, l, u)
        settings = _OSQPSettings()
        lib.osqp_set_default_settings(ctypes.byref(settings))
        settings.verbose = 0
        settings.warm_start = 1
        settings.polish = int(bool(self.polish))
        settings.max_iter = int(self.max_iter)
        settings.eps_abs = float(self.eps_abs)
        settings.eps_rel = float(self.eps_rel)
        settings.eps_prim_inf = float(self.eps_prim_inf)
        settings.eps_dual_inf = float(self.eps_dual_inf)
        work = ctypes.POINTER(_OSQPWorkspace)()
        t0 = perf_counter()
        rc = int(lib.osqp_setup(ctypes.byref(work), ctypes.byref(cdata), ctypes.byref(settings)))
        setup_wall = perf_counter() - t0
        if rc != 0 or not bool(work):
            raise RuntimeError(f"OSQP C API setup failed with code {rc}")
        keepalive = (Pp, Pi, Px, Ap, Ai, Ax, q, l, u, P_csc, A_csc, cdata, settings)
        self._state = _OSQPState(
            work=work,
            keepalive=keepalive,
            structural_hash=problem.structural_hash,
            p_values=np.asarray(P.data, dtype=np.float64).copy(),
            a_values=np.asarray(A.data, dtype=np.float64).copy(),
        )
        return lib, setup_wall

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if not self.is_available():
            raise BackendUnavailableError("verified bundled OSQP C API is unavailable")
        if not isinstance(problem, QuadraticProblem):
            raise ValueError("bundled-osqp-capi supports convex continuous QP only")
        if self.max_iter <= 0 or min(self.eps_abs, self.eps_rel, self.eps_prim_inf, self.eps_dual_inf) <= 0:
            raise ValueError("OSQP iteration/tolerance settings must be positive")

        data = _osqp_data(problem)
        reused = self._state is not None and self._state.structural_hash == problem.structural_hash
        setup_wall = 0.0
        update_wall = 0.0
        matrix_values_updated = False
        if not reused:
            lib, setup_wall = self._setup(problem, data)
            reuse_mode = "cold_capi_setup"
        else:
            paths = _bundled_paths()
            if paths is None:
                raise RuntimeError("bundled OSQP C API paths became unavailable after availability check")
            lib = _load_osqp_capi(paths)
            if self._state is None:
                raise RuntimeError("OSQP persistent state unexpectedly missing during reuse")
            q = _c_float_array(data.q); l = _c_float_array(data.l); u = _c_float_array(data.u)
            t0 = perf_counter()
            rc_q = int(lib.osqp_update_lin_cost(self._state.work, q))
            rc_b = int(lib.osqp_update_bounds(self._state.work, l, u))
            if rc_q != 0 or rc_b != 0:
                raise RuntimeError(f"OSQP C API vector update failed: q={rc_q}, bounds={rc_b}")
            p_new = np.asarray(data.P.data, dtype=np.float64)
            a_new = np.asarray(data.A.data, dtype=np.float64)
            if not np.array_equal(p_new, self._state.p_values) or not np.array_equal(a_new, self._state.a_values):
                Px = _c_float_array(p_new); Ax = _c_float_array(a_new)
                rc_m = int(
                    lib.osqp_update_P_A(
                        self._state.work,
                        Px,
                        None,
                        len(p_new),
                        Ax,
                        None,
                        len(a_new),
                    )
                )
                if rc_m != 0:
                    raise RuntimeError(f"OSQP C API matrix-value update failed: {rc_m}")
                self._state.p_values = p_new.copy()
                self._state.a_values = a_new.copy()
                matrix_values_updated = True
            update_wall = perf_counter() - t0
            reuse_mode = (
                "persistent_workspace+same_sparsity_matrix_value_update+automatic_warm_start"
                if matrix_values_updated
                else "persistent_workspace+vector_update+automatic_warm_start"
            )

        if self._state is None:
            raise RuntimeError("OSQP persistent state unexpectedly missing before solve")
        t0 = perf_counter()
        solve_rc = int(lib.osqp_solve(self._state.work))
        solve_wall = perf_counter() - t0
        if solve_rc != 0:
            raise RuntimeError(f"OSQP C API solve failed with code {solve_rc}")
        ws = self._state.work.contents
        info = ws.info.contents
        status_text = _status_text(info)
        candidate = None
        if bool(ws.solution) and bool(ws.solution.contents.x):
            arr = np.ctypeslib.as_array(ws.solution.contents.x, shape=(problem.n_variables,)).copy()
            if np.all(np.isfinite(arr)):
                candidate = arr
        has_candidate = candidate is not None
        backend_status = _osqp_status(status_text, has_candidate)

        certificate_kind = None
        certificate_check = None
        status_lower = status_text.strip().lower()
        if status_lower.startswith("primal infeasible"):
            certificate_kind = "primal_infeasibility"
            cert = None
            if bool(ws.delta_y):
                cert = np.ctypeslib.as_array(ws.delta_y, shape=(data.A.shape[0],)).copy()
            multiplier = 10.0 if status_lower.endswith("inaccurate") else 1.0
            certificate_check = _primal_certificate_check(data, cert, eps=self.eps_prim_inf * multiplier)
            if status_lower == "primal infeasible" and not certificate_check.get("valid"):
                backend_status = "infeasible_candidate"
        elif status_lower.startswith("dual infeasible"):
            certificate_kind = "dual_infeasibility"
            cert = None
            if bool(ws.delta_x):
                cert = np.ctypeslib.as_array(ws.delta_x, shape=(problem.n_variables,)).copy()
            multiplier = 10.0 if status_lower.endswith("inaccurate") else 1.0
            certificate_check = _dual_certificate_check(problem, data, cert, eps=self.eps_dual_inf * multiplier)
            if status_lower == "dual infeasible" and not certificate_check.get("valid"):
                backend_status = "unbounded_candidate"

        x = candidate if has_candidate and backend_status in {"optimal", "converged_candidate", "limit_feasible"} else None
        objective = None
        if x is not None and np.isfinite(float(info.obj_val)):
            objective = float(info.obj_val) + float(problem.linear.objective_offset)

        raw = {
            "osqp_status": status_text,
            "status_val": int(info.status_val),
            "iter": int(info.iter),
            "objective_internal": float(info.obj_val),
            "pri_res": float(info.pri_res),
            "dua_res": float(info.dua_res),
            "rho_updates": int(info.rho_updates),
            "rho_estimate": float(info.rho_estimate),
            "setup_wall_s": float(setup_wall),
            "update_wall_s": float(update_wall),
            "solve_wall_s": float(solve_wall),
            "reuse_applied": bool(reused),
            "reuse_mode": reuse_mode,
            "matrix_values_updated": bool(matrix_values_updated),
            "certificate_kind": certificate_kind,
            "certificate_check": certificate_check,
            "solver_core_direct_c_api": True,
            "factorization_reuse_claimed": False,
        }
        return BackendSolveResult(backend_status=backend_status, x=x, objective_reported=objective, raw_statistics=raw)


# ---- HiGHS C API ----
_HInt = ctypes.c_int32
_HIGHS_STATUS_OK = 0
_HIGHS_OBJ_MIN = 1
_HIGHS_OBJ_MAX = -1
_HIGHS_MATRIX_COLWISE = 1
_HIGHS_VAR_CONTINUOUS = 0
_HIGHS_VAR_INTEGER = 1
_HIGHS_MODEL_OPTIMAL = 7
_HIGHS_MODEL_INFEASIBLE = 8
_HIGHS_MODEL_UNBOUNDED_OR_INFEASIBLE = 9
_HIGHS_MODEL_UNBOUNDED = 10
_HIGHS_MODEL_OBJECTIVE_BOUND = 11
_HIGHS_MODEL_OBJECTIVE_TARGET = 12
_HIGHS_MODEL_TIME_LIMIT = 13
_HIGHS_MODEL_ITERATION_LIMIT = 14
_HIGHS_MODEL_SOLUTION_LIMIT = 16
_HIGHS_MODEL_INTERRUPT = 17


def _load_highs_capi(paths: dict[str, Path]):
    lib = ctypes.CDLL(str(paths["highs_lib"]))
    lib.Highs_version.restype = ctypes.c_char_p
    lib.Highs_create.restype = ctypes.c_void_p
    lib.Highs_destroy.argtypes = [ctypes.c_void_p]
    lib.Highs_resetGlobalScheduler.argtypes = [_HInt]
    lib.Highs_resetGlobalScheduler.restype = None
    lib.Highs_getInfinity.restype = ctypes.c_double
    lib.Highs_setBoolOptionValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _HInt]
    lib.Highs_setBoolOptionValue.restype = _HInt
    lib.Highs_setIntOptionValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, _HInt]
    lib.Highs_setIntOptionValue.restype = _HInt
    lib.Highs_setDoubleOptionValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_double]
    lib.Highs_setDoubleOptionValue.restype = _HInt
    lib.Highs_setStringOptionValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lib.Highs_setStringOptionValue.restype = _HInt
    dbl_p = ctypes.POINTER(ctypes.c_double); int_p = ctypes.POINTER(_HInt)
    lib.Highs_passLp.argtypes = [ctypes.c_void_p, _HInt, _HInt, _HInt, _HInt, _HInt, ctypes.c_double, dbl_p, dbl_p, dbl_p, dbl_p, dbl_p, int_p, int_p, dbl_p]
    lib.Highs_passLp.restype = _HInt
    lib.Highs_passMip.argtypes = [ctypes.c_void_p, _HInt, _HInt, _HInt, _HInt, _HInt, ctypes.c_double, dbl_p, dbl_p, dbl_p, dbl_p, dbl_p, int_p, int_p, dbl_p, int_p]
    lib.Highs_passMip.restype = _HInt
    lib.Highs_run.argtypes = [ctypes.c_void_p]; lib.Highs_run.restype = _HInt
    lib.Highs_zeroAllClocks.argtypes = [ctypes.c_void_p]; lib.Highs_zeroAllClocks.restype = _HInt
    lib.Highs_getRunTime.argtypes = [ctypes.c_void_p]; lib.Highs_getRunTime.restype = ctypes.c_double
    lib.Highs_getModelStatus.argtypes = [ctypes.c_void_p]; lib.Highs_getModelStatus.restype = _HInt
    lib.Highs_getObjectiveValue.argtypes = [ctypes.c_void_p]; lib.Highs_getObjectiveValue.restype = ctypes.c_double
    lib.Highs_getSolution.argtypes = [ctypes.c_void_p, dbl_p, dbl_p, dbl_p, dbl_p]; lib.Highs_getSolution.restype = _HInt
    lib.Highs_getBasis.argtypes = [ctypes.c_void_p, int_p, int_p]; lib.Highs_getBasis.restype = _HInt
    lib.Highs_setBasis.argtypes = [ctypes.c_void_p, int_p, int_p]; lib.Highs_setBasis.restype = _HInt
    lib.Highs_changeObjectiveSense.argtypes = [ctypes.c_void_p, _HInt]; lib.Highs_changeObjectiveSense.restype = _HInt
    lib.Highs_changeObjectiveOffset.argtypes = [ctypes.c_void_p, ctypes.c_double]; lib.Highs_changeObjectiveOffset.restype = _HInt
    lib.Highs_changeColsCostByRange.argtypes = [ctypes.c_void_p, _HInt, _HInt, dbl_p]; lib.Highs_changeColsCostByRange.restype = _HInt
    lib.Highs_changeColsBoundsByRange.argtypes = [ctypes.c_void_p, _HInt, _HInt, dbl_p, dbl_p]; lib.Highs_changeColsBoundsByRange.restype = _HInt
    lib.Highs_changeRowsBoundsBySet.argtypes = [ctypes.c_void_p, _HInt, int_p, dbl_p, dbl_p]; lib.Highs_changeRowsBoundsBySet.restype = _HInt
    lib.Highs_changeCoeff.argtypes = [ctypes.c_void_p, _HInt, _HInt, ctypes.c_double]; lib.Highs_changeCoeff.restype = _HInt
    lib.Highs_getIntInfoValue.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.POINTER(_HInt)]; lib.Highs_getIntInfoValue.restype = _HInt
    return lib


def _highs_status(model_status: int, has_candidate: bool) -> str:
    if model_status == _HIGHS_MODEL_OPTIMAL:
        return "optimal"
    if model_status == _HIGHS_MODEL_INFEASIBLE:
        return "infeasible"
    if model_status == _HIGHS_MODEL_UNBOUNDED:
        return "unbounded"
    if model_status == _HIGHS_MODEL_UNBOUNDED_OR_INFEASIBLE:
        return "infeasible_or_unbounded"
    if model_status in {_HIGHS_MODEL_TIME_LIMIT, _HIGHS_MODEL_ITERATION_LIMIT, _HIGHS_MODEL_SOLUTION_LIMIT, _HIGHS_MODEL_INTERRUPT, _HIGHS_MODEL_OBJECTIVE_BOUND, _HIGHS_MODEL_OBJECTIVE_TARGET}:
        return "limit_feasible" if has_candidate else "limit_no_solution"
    return "solver_error"


def _h_double(values: np.ndarray):
    arr = np.asarray(values, dtype=np.float64)
    return (ctypes.c_double * len(arr))(*arr.tolist())


def _h_int(values: np.ndarray):
    arr = np.asarray(values, dtype=np.int32)
    return (_HInt * len(arr))(*arr.tolist())


@dataclass(slots=True)
class _HighsState:
    handle: int
    structural_hash: str
    matrix_csc: sparse.csc_matrix
    effective_solver: str
    basis_col: np.ndarray | None = None
    basis_row: np.ndarray | None = None


@dataclass(slots=True)
class BundledHighsCAPIBackend:
    """Direct HiGHS C-API verification backend for CasADi's bundled library."""

    time_limit_s: float | None = None
    threads: int | None = 1
    solver: str = "simplex"
    presolve: bool = True
    enable_basis_hot_start: bool = True
    _state: _HighsState | None = field(default=None, init=False, repr=False)

    @property
    def manifest(self) -> BackendManifest:
        paths = _bundled_paths()
        solver_version = None
        metadata: dict[str, object] = {
            "native_adapter": False,
            "verification_only": True,
            "direct_c_api": True,
            "public_python_package": False,
            "bundled_by": "casadi",
            "bridge_package_version": _casadi_version(),
            "verified_bridge_package_versions": (_VERIFIED_CASADI_VERSION,),
            "expected_solver_version": _VERIFIED_HIGHS_VERSION,
            "verified_solver_versions": (_VERIFIED_HIGHS_VERSION,),
            "supports_wall_time_budget": True,
            "stateful": True,
            "reuse_semantics": "persistent HiGHS instance + numerical edits + explicit LP basis hot start",
        }
        if paths and paths["highs_lib"].is_file():
            try:
                lib = _load_highs_capi(paths)
                solver_version = (lib.Highs_version() or b"").decode("ascii", errors="replace") or None
                metadata.update({"library_sha256": _sha256(paths["highs_lib"]), "header_sha256": _sha256(paths["highs_header"]), "highs_int_bytes": ctypes.sizeof(_HInt)})
            except Exception:
                pass
        return BackendManifest(
            name="bundled-highs-capi",
            version=solver_version,
            capabilities={
                Capability.LP: SupportLevel.NATIVE,
                Capability.CONVEX_QP: SupportLevel.UNSUPPORTED,
                Capability.MILP: SupportLevel.NATIVE,
                Capability.MIP_START: SupportLevel.UNKNOWN,
                Capability.PRIMAL_START: SupportLevel.UNKNOWN,
                Capability.DUAL_START: SupportLevel.UNKNOWN,
                Capability.BASIS_START: SupportLevel.NATIVE,
                Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE,
                Capability.STRUCTURAL_INCREMENTAL_UPDATE: SupportLevel.UNSUPPORTED,
                Capability.NATIVE_REOPTIMIZATION: SupportLevel.UNSUPPORTED,
                Capability.CALLBACK_PROGRESS: SupportLevel.UNKNOWN,
                Capability.IIS: SupportLevel.UNKNOWN,
                Capability.INFEASIBILITY_CERTIFICATE: SupportLevel.UNKNOWN,
            },
            metadata=metadata,
        )

    def is_available(self) -> bool:
        if _casadi_version() != _VERIFIED_CASADI_VERSION:
            return False
        paths = _bundled_paths()
        if paths is None or not paths["highs_lib"].is_file() or not paths["highs_header"].is_file():
            return False
        try:
            lib = _load_highs_capi(paths)
            return ctypes.sizeof(_HInt) == 4 and (lib.Highs_version() or b"").decode("ascii") == _VERIFIED_HIGHS_VERSION
        except Exception:
            return False

    def close(self) -> None:
        if self._state is None:
            return
        try:
            paths = _bundled_paths()
            if paths is not None:
                _load_highs_capi(paths).Highs_destroy(ctypes.c_void_p(self._state.handle))
        finally:
            self._state = None

    def reset_global_scheduler(self, *, blocking: bool = True) -> None:
        """Release HiGHS' process-global scheduler after sequential verification.

        HiGHS documents that scheduler resources survive ``Highs_destroy`` and are
        shared across instances.  This method is intentionally separate from
        :meth:`close` because resetting the global scheduler while another HiGHS
        instance is in use is undefined behavior.  Health probes call it only after
        the backend instance has been closed and probes are executed sequentially.
        """
        paths = _bundled_paths()
        if paths is None:
            return
        _load_highs_capi(paths).Highs_resetGlobalScheduler(1 if blocking else 0)

    def __del__(self):  # pragma: no cover
        try:
            self.close()
        except Exception:
            pass

    def _set_options(self, lib, h, *, is_mip: bool) -> str:
        if int(lib.Highs_setBoolOptionValue(h, b"output_flag", 0)) < 0:
            raise RuntimeError("HiGHS failed to disable output")
        if self.threads is not None:
            if self.threads <= 0:
                raise ValueError("threads must be positive")
            if int(lib.Highs_setIntOptionValue(h, b"threads", int(self.threads))) < 0:
                raise RuntimeError("HiGHS failed to set thread count")
        if self.time_limit_s is not None:
            if self.time_limit_s <= 0:
                raise ValueError("time_limit_s must be positive")
            if int(lib.Highs_setDoubleOptionValue(h, b"time_limit", float(self.time_limit_s))) < 0:
                raise RuntimeError("HiGHS failed to set time limit")
        # HiGHS' ``solver`` option selects LP algorithms. For a MIP, forcing
        # ``simplex`` or ``ipm`` solves only the LP relaxation. Preserve integer
        # semantics by always using the MIP-capable automatic path.
        effective_solver = "choose" if is_mip else str(self.solver)
        if int(lib.Highs_setStringOptionValue(h, b"solver", effective_solver.encode())) < 0:
            raise RuntimeError("HiGHS failed to set solver")
        if int(lib.Highs_setStringOptionValue(h, b"presolve", b"on" if self.presolve else b"off")) < 0:
            raise RuntimeError("HiGHS failed to set presolve")
        return effective_solver

    def _build(self, lib, problem: LinearProblem) -> float:
        self.close()
        # HiGHS uses a process-global scheduler.  A previously executed HiGHS
        # binding (for example the CasADi verification bridge) may have created it
        # with a different thread count.  HiGHS documents that the scheduler must
        # be reset before changing the thread option.  This verification backend
        # is executed sequentially by the benchmark/health harness, so reset it
        # before constructing an independent C-API instance.
        if self.threads is not None:
            lib.Highs_resetGlobalScheduler(1)
        h = lib.Highs_create()
        if not h:
            raise RuntimeError("Highs_create returned null")
        effective_solver = self._set_options(lib, h, is_mip=problem.has_integer_variables)
        A = problem.A.tocsc(copy=True); A.sort_indices()
        cost = _h_double(problem.c); vl = _h_double(problem.variable_lower); vu = _h_double(problem.variable_upper)
        rl = _h_double(problem.constraint_lower); ru = _h_double(problem.constraint_upper)
        starts = _h_int(A.indptr); indices = _h_int(A.indices); values = _h_double(A.data)
        sense = _HIGHS_OBJ_MIN if problem.objective_sense is ObjectiveSense.MINIMIZE else _HIGHS_OBJ_MAX
        t0 = perf_counter()
        if problem.has_integer_variables:
            integrality = _h_int(np.array([_HIGHS_VAR_CONTINUOUS if d == VariableDomain.CONTINUOUS.value else _HIGHS_VAR_INTEGER for d in problem.domains], dtype=np.int32))
            rc = int(lib.Highs_passMip(h, problem.n_variables, problem.n_constraints, problem.nnz, _HIGHS_MATRIX_COLWISE, sense, float(problem.objective_offset), cost, vl, vu, rl, ru, starts, indices, values, integrality))
        else:
            rc = int(lib.Highs_passLp(h, problem.n_variables, problem.n_constraints, problem.nnz, _HIGHS_MATRIX_COLWISE, sense, float(problem.objective_offset), cost, vl, vu, rl, ru, starts, indices, values))
        build_wall = perf_counter() - t0
        if rc != _HIGHS_STATUS_OK:
            lib.Highs_destroy(h)
            raise RuntimeError(f"HiGHS C API model load failed with code {rc}")
        self._state = _HighsState(handle=int(h), structural_hash=problem.structural_hash, matrix_csc=A.copy(), effective_solver=effective_solver)
        return build_wall

    def _update(self, lib, problem: LinearProblem) -> tuple[float, bool, int]:
        if self._state is None:
            raise RuntimeError("HiGHS persistent state unexpectedly missing during update")
        h = ctypes.c_void_p(self._state.handle)
        t0 = perf_counter()
        n, m = problem.n_variables, problem.n_constraints
        sense = _HIGHS_OBJ_MIN if problem.objective_sense is ObjectiveSense.MINIMIZE else _HIGHS_OBJ_MAX
        calls = [
            int(lib.Highs_changeObjectiveSense(h, sense)),
            int(lib.Highs_changeObjectiveOffset(h, float(problem.objective_offset))),
        ]
        if n:
            calls.append(int(lib.Highs_changeColsCostByRange(h, 0, n - 1, _h_double(problem.c))))
            calls.append(int(lib.Highs_changeColsBoundsByRange(h, 0, n - 1, _h_double(problem.variable_lower), _h_double(problem.variable_upper))))
        if m:
            row_idx = _h_int(np.arange(m, dtype=np.int32))
            calls.append(int(lib.Highs_changeRowsBoundsBySet(h, m, row_idx, _h_double(problem.constraint_lower), _h_double(problem.constraint_upper))))
        A_new = problem.A.tocsc(copy=True); A_new.sort_indices()
        changed = 0
        if A_new.shape != self._state.matrix_csc.shape or not np.array_equal(A_new.indptr, self._state.matrix_csc.indptr) or not np.array_equal(A_new.indices, self._state.matrix_csc.indices):
            raise RuntimeError("same structural hash produced different sparse structure")
        diff = np.flatnonzero(A_new.data != self._state.matrix_csc.data)
        if len(diff):
            # Convert CSC positions to (row,col) only for values that changed.
            for col in range(A_new.shape[1]):
                lo, hi = A_new.indptr[col], A_new.indptr[col + 1]
                for pos in diff[(diff >= lo) & (diff < hi)]:
                    rc = int(lib.Highs_changeCoeff(h, int(A_new.indices[pos]), col, float(A_new.data[pos])))
                    calls.append(rc); changed += 1
            self._state.matrix_csc = A_new.copy()
        if any(rc < 0 for rc in calls):
            raise RuntimeError(f"HiGHS C API numerical update failed: {calls}")
        basis_applied = False
        if self.enable_basis_hot_start and not problem.has_integer_variables and self._state.basis_col is not None and self._state.basis_row is not None:
            rc = int(lib.Highs_setBasis(h, _h_int(self._state.basis_col), _h_int(self._state.basis_row)))
            if rc == _HIGHS_STATUS_OK:
                basis_applied = True
        return perf_counter() - t0, basis_applied, changed

    def solve(self, problem: LinearProblem | QuadraticProblem) -> BackendSolveResult:
        if isinstance(problem, QuadraticProblem):
            require_confirmed_convexity(problem)
        if not self.is_available():
            raise BackendUnavailableError("verified bundled HiGHS C API is unavailable")
        if not isinstance(problem, LinearProblem):
            raise ValueError("bundled-highs-capi M13 supports LP/MILP only; QP remains verified through other paths")
        paths = _bundled_paths()
        if paths is None:
            raise RuntimeError("bundled HiGHS C API paths became unavailable after availability check")
        lib = _load_highs_capi(paths)
        reused = self._state is not None and self._state.structural_hash == problem.structural_hash
        build_wall = 0.0; update_wall = 0.0; basis_applied = False; matrix_values_changed = 0
        if not reused:
            build_wall = self._build(lib, problem)
            reuse_mode = "cold_capi_model_build"
        else:
            update_wall, basis_applied, matrix_values_changed = self._update(lib, problem)
            reuse_mode = "persistent_highs_model+numeric_update"
            if basis_applied:
                reuse_mode += "+explicit_basis_hot_start"
        if self._state is None:
            raise RuntimeError("HiGHS persistent state unexpectedly missing before solve")
        h = ctypes.c_void_p(self._state.handle)
        lib.Highs_zeroAllClocks(h)
        t0 = perf_counter(); run_rc = int(lib.Highs_run(h)); solve_wall = perf_counter() - t0
        if run_rc < 0:
            raise RuntimeError(f"HiGHS C API solve failed with code {run_rc}")
        model_status = int(lib.Highs_getModelStatus(h))
        x_arr = (ctypes.c_double * problem.n_variables)()
        col_dual = (ctypes.c_double * problem.n_variables)()
        row_val = (ctypes.c_double * problem.n_constraints)()
        row_dual = (ctypes.c_double * problem.n_constraints)()
        sol_rc = int(lib.Highs_getSolution(h, x_arr, col_dual, row_val, row_dual))
        candidate = None
        if sol_rc == _HIGHS_STATUS_OK:
            arr = np.ctypeslib.as_array(x_arr).copy()
            if arr.shape == (problem.n_variables,) and np.all(np.isfinite(arr)):
                candidate = arr
        backend_status = _highs_status(model_status, candidate is not None)
        x = candidate if candidate is not None and backend_status in {"optimal", "limit_feasible"} else None
        objective = float(problem.c @ x + problem.objective_offset) if x is not None else None

        simplex_iters = _HInt(0); ipm_iters = _HInt(0)
        lib.Highs_getIntInfoValue(h, b"simplex_iteration_count", ctypes.byref(simplex_iters))
        lib.Highs_getIntInfoValue(h, b"ipm_iteration_count", ctypes.byref(ipm_iters))
        basis_captured = False
        if self.enable_basis_hot_start and not problem.has_integer_variables and backend_status == "optimal":
            col_status = (_HInt * problem.n_variables)(); row_status = (_HInt * problem.n_constraints)()
            if int(lib.Highs_getBasis(h, col_status, row_status)) == _HIGHS_STATUS_OK:
                self._state.basis_col = np.ctypeslib.as_array(col_status).copy().astype(np.int32)
                self._state.basis_row = np.ctypeslib.as_array(row_status).copy().astype(np.int32)
                basis_captured = True
        raw = {
            "model_status": model_status,
            "objective_internal": float(lib.Highs_getObjectiveValue(h)),
            "simplex_iteration_count": int(simplex_iters.value),
            "ipm_iteration_count": int(ipm_iters.value),
            "highs_run_time_s": float(lib.Highs_getRunTime(h)),
            "build_wall_s": float(build_wall),
            "update_wall_s": float(update_wall),
            "solve_wall_s": float(solve_wall),
            "reuse_applied": bool(reused),
            "reuse_mode": reuse_mode,
            "basis_applied": bool(basis_applied),
            "basis_captured": bool(basis_captured),
            "matrix_values_changed": int(matrix_values_changed),
            "solver_core_direct_c_api": True,
            "requested_solver": str(self.solver),
            "effective_solver": self._state.effective_solver,
        }
        return BackendSolveResult(backend_status=backend_status, x=x, objective_reported=objective, raw_statistics=raw)
