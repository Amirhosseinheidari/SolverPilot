from __future__ import annotations

from dataclasses import dataclass
from datetime import date
import importlib.metadata
from typing import Iterable

from solverpilot.backends.health import BackendProbeStatus, probe_backend

from .v2 import (
    BackendCapabilityManifestV2,
    CapabilityEvidence,
    CapabilityKey,
    project_legacy_manifest,
    with_verified_claims,
)


@dataclass(frozen=True, slots=True)
class ConformanceCheck:
    capability: CapabilityKey
    passed: bool
    evidence_id: str
    detail: str


@dataclass(frozen=True, slots=True)
class BackendConformanceReport:
    backend: str
    available: bool
    checks: tuple[ConformanceCheck, ...]
    manifest: BackendCapabilityManifestV2

    @property
    def passed(self) -> bool:
        return self.available and all(check.passed for check in self.checks)


def _package_version(name: str) -> str | None:
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None


def _binding_identity(backend) -> tuple[str, str | None]:
    module = type(backend).__module__
    if module.startswith("solverpilot.backends.scipy_"):
        return "scipy", _package_version("scipy")
    if module.endswith("nlopt_native"):
        return "nlopt", _package_version("nlopt")
    if module.endswith("osqp_native"):
        return "osqp", _package_version("osqp")
    if module.endswith("highspy_native"):
        return "highspy", _package_version("highspy")
    if module.endswith("pyscipopt_native"):
        return "pyscipopt", _package_version("pyscipopt")
    if module.endswith("casadi_conic") or module.endswith("bundled_capi"):
        return "casadi", _package_version("casadi")
    return module, None


def _adapter_version() -> str | None:
    # Prefer the imported source version so source-tree conformance cannot silently bind
    # evidence to an older wheel installed in the same environment.
    try:
        import solverpilot
        return solverpilot.__version__
    except Exception:
        return _package_version("solverpilot")


def conform_backend_problem_classes(backend) -> BackendConformanceReport:
    """Verify currently exposed solve/result channels against native smoke problems.

    P3 deliberately verifies only what the current adapter surface can exercise without
    inventing a new stateful/callback API. Advanced legacy declarations remain projected
    and are therefore unusable by a verified-only P4 compiler until their own conformance
    tests are added in the milestone that exposes those adapter methods.
    """

    binding, binding_version = _binding_identity(backend)
    adapter_version = _adapter_version()
    projected = project_legacy_manifest(
        backend.manifest,
        binding=binding,
        binding_version=binding_version,
        adapter_version=adapter_version,
    )
    if not backend.is_available():
        return BackendConformanceReport(
            backend=backend.manifest.name,
            available=False,
            checks=(),
            manifest=projected,
        )

    health = probe_backend(backend)
    verified = {}
    checks: list[ConformanceCheck] = []
    capability_map = {
        "lp": CapabilityKey.PROBLEM_LP,
        "milp": CapabilityKey.PROBLEM_MILP,
        "convex_qp": CapabilityKey.PROBLEM_CONVEX_QP,
    }
    today = date.today().isoformat()
    for check in health.checks:
        key = capability_map.get(check.capability.value)
        if key is None:
            continue
        evidence_id = f"p3.native-smoke:{backend.manifest.name}:{check.capability.value}"
        passed = bool(check.passed and check.validation_valid)
        checks.append(
            ConformanceCheck(
                capability=key,
                passed=passed,
                evidence_id=evidence_id,
                detail=(
                    "native smoke solved and original-space validation passed"
                    if passed
                    else f"native smoke failed: {check.error or check.backend_status}"
                ),
            )
        )
        if passed:
            ev = CapabilityEvidence(
                evidence_id=evidence_id,
                kind="native_solve_validation",
                backend_versions=(backend.manifest.version,) if backend.manifest.version else (),
                binding_versions=(binding_version,) if binding_version else (),
                adapter_versions=(adapter_version,) if adapter_version else (),
                verified_on=today,
                notes=("smoke result independently validated in canonical problem space",),
            )
            verified[key] = ev
            # The current BackendSolveResult surface demonstrably returns these artifacts.
            verified.setdefault(CapabilityKey.RESULT_PRIMAL, ev)
            verified.setdefault(CapabilityKey.RESULT_OBJECTIVE, ev)
            verified.setdefault(CapabilityKey.CONSTRAINT_LINEAR, ev)

    manifest = with_verified_claims(projected, verified)
    return BackendConformanceReport(
        backend=backend.manifest.name,
        available=True,
        checks=tuple(checks),
        manifest=manifest,
    )


def conform_backends(backends: Iterable[object]) -> tuple[BackendConformanceReport, ...]:
    return tuple(conform_backend_problem_classes(backend) for backend in backends)
