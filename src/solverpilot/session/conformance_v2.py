from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import importlib.metadata
from typing import Any

import numpy as np
from scipy import sparse

from solverpilot.capabilities import (
    BackendCapabilityManifestV2,
    CapabilityEvidence,
    CapabilityKey,
    CapabilityStatus,
    project_legacy_manifest,
    with_verified_claims,
)
from solverpilot.problem import LinearProblem, QuadraticProblem
from solverpilot.validate import CandidateSolution, validate_solution


@dataclass(frozen=True, slots=True)
class PersistenceConformanceCheck:
    capability: CapabilityKey
    passed: bool
    detail: str
    evidence_id: str


@dataclass(frozen=True, slots=True)
class PersistenceConformanceReport:
    backend: str
    available: bool
    checks: tuple[PersistenceConformanceCheck, ...]
    manifest: BackendCapabilityManifestV2

    @property
    def passed(self) -> bool:
        return self.available and bool(self.checks) and all(check.passed for check in self.checks)


def _pkg(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _adapter_version() -> str | None:
    try:
        import solverpilot
        return solverpilot.__version__
    except Exception:
        return None


def _binding(backend) -> tuple[str, str | None]:
    module = type(backend).__module__
    if module.endswith("bundled_capi") or module.endswith("casadi_conic"):
        return "casadi", _pkg("casadi")
    if module.endswith("highspy_native"):
        return "highspy", _pkg("highspy")
    if module.endswith("osqp_native"):
        return "osqp", _pkg("osqp")
    if module.startswith("solverpilot.backends.scipy_"):
        return "scipy", _pkg("scipy")
    return module, None


def _ev(backend, binding_version: str | None, key: CapabilityKey) -> CapabilityEvidence:
    return CapabilityEvidence(
        evidence_id=f"p5.persistence:{backend.manifest.name}:{key.value}",
        kind="persistent_mutation_native_validation",
        backend_versions=(backend.manifest.version,) if backend.manifest.version else (),
        binding_versions=(binding_version,) if binding_version else (),
        adapter_versions=(_adapter_version(),) if _adapter_version() else (),
        verified_on=date.today().isoformat(),
        notes=("same backend instance reused; native adapter reported reuse; candidate independently validated",),
    )


def _lp_base() -> LinearProblem:
    return LinearProblem.from_data(
        A=[[1.0, 1.0], [1.0, 0.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[5.0, 5.0],
        constraint_lower=[1.0, -np.inf], constraint_upper=[np.inf, 4.0],
    )


def _qp_base() -> QuadraticProblem:
    return QuadraticProblem.from_data(
        P=[[2.0, 0.0], [0.0, 2.0]], A=[[1.0, 1.0]], q=[-2.0, -4.0],
        variable_lower=[-5.0, -5.0], variable_upper=[5.0, 5.0],
        constraint_lower=[-np.inf], constraint_upper=[10.0], verify_convexity=True,
    )


def _variants(problem):
    if isinstance(problem, LinearProblem):
        yield CapabilityKey.INCREMENTAL_OBJECTIVE, LinearProblem.from_data(
            A=problem.A, c=[2.0, 1.0], variable_lower=problem.variable_lower,
            variable_upper=problem.variable_upper, constraint_lower=problem.constraint_lower,
            constraint_upper=problem.constraint_upper,
        )
        yield CapabilityKey.INCREMENTAL_RHS, LinearProblem.from_data(
            A=problem.A, c=problem.c, variable_lower=problem.variable_lower,
            variable_upper=problem.variable_upper, constraint_lower=[1.5, -np.inf],
            constraint_upper=problem.constraint_upper,
        )
        yield CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS, LinearProblem.from_data(
            A=problem.A, c=problem.c, variable_lower=[0.0, 0.0], variable_upper=[3.0, 5.0],
            constraint_lower=problem.constraint_lower, constraint_upper=problem.constraint_upper,
        )
        A = problem.A.copy().tolil(); A[0, 0] = 1.5
        yield CapabilityKey.INCREMENTAL_MATRIX_VALUES, LinearProblem.from_data(
            A=A.tocsr(), c=problem.c, variable_lower=problem.variable_lower,
            variable_upper=problem.variable_upper, constraint_lower=problem.constraint_lower,
            constraint_upper=problem.constraint_upper,
        )
    else:
        lin = problem.linear
        yield CapabilityKey.INCREMENTAL_OBJECTIVE, QuadraticProblem.from_data(
            P=problem.P, A=lin.A, q=[-3.0, -4.0], variable_lower=lin.variable_lower,
            variable_upper=lin.variable_upper, constraint_lower=lin.constraint_lower,
            constraint_upper=lin.constraint_upper, verify_convexity=True,
        )
        yield CapabilityKey.INCREMENTAL_RHS, QuadraticProblem.from_data(
            P=problem.P, A=lin.A, q=lin.c, variable_lower=lin.variable_lower,
            variable_upper=lin.variable_upper, constraint_lower=[-np.inf],
            constraint_upper=[8.0], verify_convexity=True,
        )
        yield CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS, QuadraticProblem.from_data(
            P=problem.P, A=lin.A, q=lin.c, variable_lower=[-3.0, -5.0],
            variable_upper=lin.variable_upper, constraint_lower=lin.constraint_lower,
            constraint_upper=lin.constraint_upper, verify_convexity=True,
        )
        A = lin.A.copy().tolil(); A[0, 0] = 1.5
        P = problem.P.copy().tolil(); P[0, 0] = 3.0
        yield CapabilityKey.INCREMENTAL_MATRIX_VALUES, QuadraticProblem.from_data(
            P=P.tocsr(), A=A.tocsr(), q=lin.c, variable_lower=lin.variable_lower,
            variable_upper=lin.variable_upper, constraint_lower=lin.constraint_lower,
            constraint_upper=lin.constraint_upper, verify_convexity=True,
        )


def conform_persistent_backend(backend) -> PersistenceConformanceReport:
    binding, binding_version = _binding(backend)
    base_manifest = project_legacy_manifest(
        backend.manifest,
        binding=binding,
        binding_version=binding_version,
        adapter_version=_adapter_version(),
    )
    if not backend.is_available():
        return PersistenceConformanceReport(backend.manifest.name, False, (), base_manifest)

    problem = None
    if backend.manifest.name == "bundled-highs-capi":
        problem = _lp_base()
    elif backend.manifest.name == "bundled-osqp-capi":
        problem = _qp_base()
    else:
        return PersistenceConformanceReport(backend.manifest.name, True, (), base_manifest)

    checks: list[PersistenceConformanceCheck] = []
    verified: dict[CapabilityKey, CapabilityEvidence] = {}
    overrides: dict[CapabilityKey, CapabilityStatus] = {}
    restriction_overrides: dict[CapabilityKey, dict[str, Any]] = {}

    try:
        first = backend.solve(problem)
        valid0 = first.x is not None and validate_solution(
            problem, CandidateSolution(x=first.x, objective_reported=first.objective_reported)
        ).valid
        cold_ok = bool(valid0 and not bool((first.raw_statistics or {}).get("reuse_applied", False)))
        if not cold_ok:
            return PersistenceConformanceReport(backend.manifest.name, True, (), base_manifest)

        all_updates = True
        for key, variant in _variants(problem):
            # Rebase before each operation so every granular claim is independently exercised.
            close = getattr(backend, "close", None)
            if callable(close):
                close()
            base_result = backend.solve(problem)
            result = backend.solve(variant)
            raw = result.raw_statistics or {}
            valid = result.x is not None and validate_solution(
                variant, CandidateSolution(x=result.x, objective_reported=result.objective_reported)
            ).valid
            passed = bool(valid and raw.get("reuse_applied") is True)
            if key is CapabilityKey.INCREMENTAL_MATRIX_VALUES:
                passed = passed and (
                    bool(raw.get("matrix_values_updated"))
                    or int(raw.get("matrix_values_changed", 0)) > 0
                )
            evidence_id = f"p5.persistence:{backend.manifest.name}:{key.value}"
            checks.append(PersistenceConformanceCheck(key, passed, "native same-instance mutation + independent validation", evidence_id))
            all_updates = all_updates and passed
            if passed:
                verified[key] = _ev(backend, binding_version, key)
                overrides[key] = CapabilityStatus.SUPPORTED
                restriction_overrides[key] = {}

        persistent_ok = all_updates
        key = CapabilityKey.LIFECYCLE_PERSISTENT
        checks.append(PersistenceConformanceCheck(key, persistent_ok, "same backend instance preserved across all verified numeric mutations", f"p5.persistence:{backend.manifest.name}:{key.value}"))
        if persistent_ok:
            verified[key] = _ev(backend, binding_version, key)
            overrides[key] = CapabilityStatus.SUPPORTED
            restriction_overrides[key] = {}

        if backend.manifest.name == "bundled-highs-capi":
            close = getattr(backend, "close", None)
            if callable(close): close()
            backend.solve(problem)
            changed = next(v for k, v in _variants(problem) if k is CapabilityKey.INCREMENTAL_OBJECTIVE)
            second = backend.solve(changed)
            basis_ok = bool((second.raw_statistics or {}).get("basis_applied") is True)
            key = CapabilityKey.START_BASIS
            checks.append(PersistenceConformanceCheck(key, basis_ok, "captured LP basis was applied on second solve", f"p5.persistence:{backend.manifest.name}:{key.value}"))
            if basis_ok:
                verified[key] = _ev(backend, binding_version, key)
                overrides[key] = CapabilityStatus.SUPPORTED
                restriction_overrides[key] = {}
    finally:
        close = getattr(backend, "close", None)
        if callable(close):
            try: close()
            except Exception: pass

    manifest = with_verified_claims(
        base_manifest,
        verified,
        status_overrides=overrides,
        restriction_overrides=restriction_overrides,
    )
    return PersistenceConformanceReport(backend.manifest.name, True, tuple(checks), manifest)
