from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from time import perf_counter
from typing import Iterable

import numpy as np
from scipy import sparse

from solverpilot.capabilities import Capability, SupportLevel
from solverpilot.problem import LinearProblem, ObjectiveSense, QuadraticProblem, VariableDomain

from .base import Backend


class BackendProbeStatus(str, Enum):
    HEALTHY = "healthy"
    UNHEALTHY = "unhealthy"
    UNAVAILABLE = "unavailable"
    NO_SMOKE_CAPABILITY = "no_smoke_capability"


@dataclass(frozen=True, slots=True)
class BackendProbeCheck:
    capability: Capability
    passed: bool
    public_status: str | None
    validation_valid: bool | None
    objective: float | None
    expected_objective: float
    objective_abs_error: float | None
    duration_s: float
    error: str | None = None


@dataclass(frozen=True, slots=True)
class BackendHealthReport:
    backend: str
    version: str | None
    available: bool
    status: BackendProbeStatus
    checks: tuple[BackendProbeCheck, ...]
    metadata: dict[str, object]

    @property
    def healthy(self) -> bool:
        return self.status is BackendProbeStatus.HEALTHY


def _supports_smoke(manifest, capability: Capability) -> bool:
    return manifest.support(capability) in {SupportLevel.NATIVE, SupportLevel.EMULATED_SAFE}


def _lp_smoke() -> tuple[LinearProblem, float]:
    # min x, 1 <= x <= 2 -> optimum 1
    problem = LinearProblem.from_data(
        A=[[1.0]],
        c=[1.0],
        variable_lower=[0.0],
        variable_upper=[2.0],
        constraint_lower=[1.0],
        constraint_upper=[np.inf],
        domains=[VariableDomain.CONTINUOUS],
        name="solverpilot_backend_probe_lp",
    )
    return problem, 1.0


def _milp_smoke() -> tuple[LinearProblem, float]:
    # max x + 2y, x+y <= 1, x,y binary -> optimum 2
    problem = LinearProblem.from_data(
        A=[[1.0, 1.0]],
        c=[1.0, 2.0],
        variable_lower=[0.0, 0.0],
        variable_upper=[1.0, 1.0],
        constraint_lower=[-np.inf],
        constraint_upper=[1.0],
        domains=[VariableDomain.BINARY, VariableDomain.BINARY],
        objective_sense=ObjectiveSense.MAXIMIZE,
        name="solverpilot_backend_probe_milp",
    )
    return problem, 2.0


# QuadraticProblem.from_data deliberately has no name parameter. Keep the smoke factory
# separate from public metadata so backend probing never relies on optional labels.
def _qp_smoke() -> tuple[QuadraticProblem, float]:  # type: ignore[no-redef]
    problem = QuadraticProblem.from_data(
        P=[[2.0]],
        A=sparse.csr_matrix((0, 1), dtype=np.float64),
        q=[-4.0],
        variable_lower=[0.0],
        variable_upper=[3.0],
        constraint_lower=[],
        constraint_upper=[],
    )
    return problem, -4.0


def _check_one(
    backend: Backend,
    capability: Capability,
    *,
    atol: float,
    rtol: float,
) -> BackendProbeCheck:
    if capability is Capability.LP:
        problem, expected = _lp_smoke()
    elif capability is Capability.MILP:
        problem, expected = _milp_smoke()
    elif capability is Capability.CONVEX_QP:
        problem, expected = _qp_smoke()
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"unsupported smoke capability: {capability}")

    # Lazy import avoids a backends -> runtime -> backends import cycle.
    from solverpilot.runtime.executor import execute

    t0 = perf_counter()
    try:
        result = execute(problem, backend)
        duration = perf_counter() - t0
        valid = bool(result.validation and result.validation.valid)
        objective = None if result.objective is None else float(result.objective)
        error = None if objective is None else abs(objective - expected)
        passed = bool(
            valid
            and objective is not None
            and np.isclose(objective, expected, atol=atol, rtol=rtol)
        )
        return BackendProbeCheck(
            capability=capability,
            passed=passed,
            public_status=result.status.value,
            validation_valid=None if result.validation is None else bool(result.validation.valid),
            objective=objective,
            expected_objective=expected,
            objective_abs_error=error,
            duration_s=duration,
            error=None if passed else "smoke solve did not match validated expected objective",
        )
    except Exception as exc:
        return BackendProbeCheck(
            capability=capability,
            passed=False,
            public_status=None,
            validation_valid=None,
            objective=None,
            expected_objective=expected,
            objective_abs_error=None,
            duration_s=perf_counter() - t0,
            error=f"{type(exc).__name__}: {exc}",
        )


def probe_backend(
    backend: Backend,
    *,
    atol: float = 1e-6,
    rtol: float = 1e-7,
) -> BackendHealthReport:
    """Actively verify an adapter on tiny canonical problems.

    ``is_available()`` only proves that an import path exists. This probe additionally
    asks every supported core problem class (LP, MILP, convex QP) to produce a solution
    that passes SolverPilot's independent validator and matches a known objective.

    Probing is explicit because it executes solvers and may initialize native libraries.
    It is therefore never run implicitly inside :func:`solverpilot.solve`.
    """

    manifest = backend.manifest
    metadata = dict(manifest.metadata)
    if not backend.is_available():
        return BackendHealthReport(
            backend=manifest.name,
            version=manifest.version,
            available=False,
            status=BackendProbeStatus.UNAVAILABLE,
            checks=(),
            metadata=metadata,
        )

    capabilities = tuple(
        cap
        for cap in (Capability.LP, Capability.MILP, Capability.CONVEX_QP)
        if _supports_smoke(manifest, cap)
    )
    if not capabilities:
        return BackendHealthReport(
            backend=manifest.name,
            version=manifest.version,
            available=True,
            status=BackendProbeStatus.NO_SMOKE_CAPABILITY,
            checks=(),
            metadata=metadata,
        )

    cleanup_error: str | None = None
    try:
        checks = tuple(_check_one(backend, cap, atol=atol, rtol=rtol) for cap in capabilities)
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try:
                close()
                reset_scheduler = getattr(backend, "reset_global_scheduler", None)
                if callable(reset_scheduler):
                    # Health probes are sequential.  HiGHS documents that its global
                    # scheduler survives Highs_destroy and must be reset to release
                    # all scheduler resources between independent verification runs.
                    reset_scheduler(blocking=True)
            except Exception as exc:  # cleanup failures are part of backend health
                cleanup_error = f"{type(exc).__name__}: {exc}"

    if cleanup_error is not None:
        metadata["cleanup_error"] = cleanup_error
    status = (
        BackendProbeStatus.HEALTHY
        if all(check.passed for check in checks) and cleanup_error is None
        else BackendProbeStatus.UNHEALTHY
    )
    return BackendHealthReport(
        backend=manifest.name,
        version=manifest.version,
        available=True,
        status=status,
        checks=checks,
        metadata=metadata,
    )


def probe_backends(
    backends: Iterable[Backend],
    *,
    atol: float = 1e-6,
    rtol: float = 1e-7,
) -> tuple[BackendHealthReport, ...]:
    return tuple(probe_backend(b, atol=atol, rtol=rtol) for b in backends)
