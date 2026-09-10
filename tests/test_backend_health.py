from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from solverpilot import builtin_backend_candidates
from solverpilot.backends import (
    BackendProbeStatus,
    BackendSolveResult,
    ScipyHighsBackend,
    ScipyHighsLPBackend,
    ScipySLSQPQPBackend,
    NLoptNativeBackend,
    probe_backend,
    probe_backends,
)
from solverpilot.capabilities import BackendManifest, Capability, SupportLevel


def test_probe_real_base_backends_are_healthy():
    backends = [
        ScipyHighsLPBackend(method="highs-ds"),
        ScipyHighsLPBackend(method="highs-ipm"),
        ScipyHighsBackend(),
        ScipySLSQPQPBackend(),
    ]
    reports = probe_backends(backends)
    assert len(reports) == 4
    assert all(report.status is BackendProbeStatus.HEALTHY for report in reports)
    by_name = {r.backend: r for r in reports}
    assert [c.capability.value for c in by_name["scipy-highs-bridge"].checks] == ["lp", "milp"]
    assert [c.capability.value for c in by_name["scipy-slsqp-qp-bridge"].checks] == ["convex_qp"]
    assert all(c.validation_valid for r in reports for c in r.checks)


def test_probe_optional_nlopt_backend():
    import pytest
    backend = NLoptNativeBackend()
    if not backend.is_available():
        pytest.skip("optional NLopt is not installed")
    report = probe_backend(backend)
    assert report.status is BackendProbeStatus.HEALTHY
    assert [c.capability.value for c in report.checks] == ["lp", "convex_qp"]


def test_builtin_candidates_surface_unavailable_optional_adapters():
    candidates = builtin_backend_candidates()
    names = {b.manifest.name for b in candidates}
    assert names == {
        "highspy-native",
        "osqp-native",
        "pyscipopt-native",
        "scipy-highs-ds",
        "scipy-highs-ipm",
        "scipy-highs-bridge",
        "scipy-slsqp-qp-bridge",
        "nlopt-slsqp-native",
        "casadi-osqp-bridge",
        "casadi-highs-bridge",
        "casadi-cbc-bridge",
        "bundled-osqp-capi",
        "bundled-highs-capi",
    }
    reports = probe_backends(candidates)
    by_name = {r.backend: r for r in reports}
    # These assertions describe the current CI/runtime environment, not solver support in general.
    for name in ("highspy-native", "osqp-native", "pyscipopt-native"):
        assert by_name[name].status in {BackendProbeStatus.UNAVAILABLE, BackendProbeStatus.HEALTHY}


@dataclass(slots=True)
class InvalidLPBackend:
    @property
    def manifest(self):
        return BackendManifest(
            name="invalid-lp",
            version="test",
            capabilities={Capability.LP: SupportLevel.NATIVE},
        )

    def is_available(self) -> bool:
        return True

    def solve(self, problem):
        # x=0 violates the smoke problem's x>=1 constraint.
        return BackendSolveResult("optimal", np.array([0.0]), 0.0, {})


def test_probe_detects_available_but_invalid_backend():
    report = probe_backend(InvalidLPBackend())
    assert report.available
    assert report.status is BackendProbeStatus.UNHEALTHY
    assert len(report.checks) == 1
    assert not report.checks[0].passed
    assert report.checks[0].validation_valid is False


@dataclass(slots=True)
class ClosableLPBackend:
    closed: bool = False

    @property
    def manifest(self):
        return BackendManifest(
            name="closable-lp",
            version="test",
            capabilities={Capability.LP: SupportLevel.NATIVE},
        )

    def is_available(self) -> bool:
        return True

    def solve(self, problem):
        return BackendSolveResult("optimal", np.array([1.0]), 1.0, {})

    def close(self) -> None:
        self.closed = True


def test_probe_closes_stateful_backend_after_health_check():
    backend = ClosableLPBackend()
    report = probe_backend(backend)
    assert report.status is BackendProbeStatus.HEALTHY
    assert backend.closed is True
