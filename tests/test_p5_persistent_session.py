import numpy as np
import pytest

pytest.importorskip("casadi", reason="optional CasADi extra not installed in core CI")

from solverpilot import (
    Model,
    MutationExecutionPath,
    PersistentSession,
    PublicStatus,
    conform_persistent_backend,
)
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend, ScipyHighsBackend
from solverpilot.capabilities import CapabilityKey


def _highs_binding():
    backend = BundledHighsCAPIBackend(threads=1, solver="simplex")
    assert backend.is_available(), "P5 release requires verified bundled HiGHS C API"
    report = conform_persistent_backend(backend)
    assert report.passed
    return backend, report.manifest


def _osqp_binding():
    backend = BundledOSQPCAPIBackend()
    assert backend.is_available(), "P5 release requires verified bundled OSQP C API"
    report = conform_persistent_backend(backend)
    assert report.passed
    return backend, report.manifest


def test_highs_persistence_conformance_verifies_granular_mutations_and_basis():
    backend, manifest = _highs_binding()
    for key in (
        CapabilityKey.LIFECYCLE_PERSISTENT,
        CapabilityKey.INCREMENTAL_OBJECTIVE,
        CapabilityKey.INCREMENTAL_RHS,
        CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS,
        CapabilityKey.INCREMENTAL_MATRIX_VALUES,
        CapabilityKey.START_BASIS,
    ):
        assert manifest.supports_verified(key), key
    backend.close()


def test_osqp_persistence_conformance_verifies_granular_mutations():
    backend, manifest = _osqp_binding()
    for key in (
        CapabilityKey.LIFECYCLE_PERSISTENT,
        CapabilityKey.INCREMENTAL_OBJECTIVE,
        CapabilityKey.INCREMENTAL_RHS,
        CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS,
        CapabilityKey.INCREMENTAL_MATRIX_VALUES,
    ):
        assert manifest.supports_verified(key), key
    # Automatic warm start is not silently upgraded to explicit start channels.
    assert not manifest.supports_verified(CapabilityKey.START_PRIMAL)
    assert not manifest.supports_verified(CapabilityKey.START_DUAL)
    backend.close()


def test_persistent_highs_objective_parameter_uses_native_patch_and_validates_original():
    m = Model("p5-lp")
    x = m.variable(2, lower=0.0, upper=5.0, name="x")
    c = m.parameter(2, value=[1.0, 2.0], name="c")
    m.add(x.sum() >= 1.0)
    m.minimize(c @ x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        r0 = s.solve()
        assert r0.status is PublicStatus.VALID_OPTIMAL
        assert r0.trace.path is MutationExecutionPath.COLD_BUILD
        assert r0.trace.original_validation_valid
        s.set_parameter(c, [3.0, 1.0])
        r1 = s.solve()
        assert r1.status is PublicStatus.VALID_OPTIMAL
        assert r1.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r1.trace.backend_reuse_applied is True
        assert "objective_vector" in [k.value for k in r1.trace.decision.mutation_kinds]
        assert r1.trace.original_validation_valid
        assert r1.objective == pytest.approx(1.0)
        assert s.revision == 1


def test_persistent_highs_rhs_parameter_uses_native_patch():
    m = Model("p5-rhs")
    x = m.variable(2, lower=0.0, upper=10.0)
    rhs = m.parameter(value=1.0, name="rhs")
    m.add(x.sum() >= rhs)
    m.minimize(x.sum())
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        assert s.solve().objective == pytest.approx(1.0)
        rhs.value = 2.5
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r.trace.backend_reuse_applied is True
        assert r.objective == pytest.approx(2.5)


def test_persistent_highs_matrix_value_parameter_uses_native_patch():
    m = Model("p5-matrix")
    x = m.variable(lower=0.0, upper=10.0)
    a = m.parameter(value=1.0, name="a")
    m.add(a * x >= 2.0)
    m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        assert s.solve().objective == pytest.approx(2.0)
        a.value = 2.0
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r.trace.backend_reuse_applied is True
        assert r.objective == pytest.approx(1.0)


def test_parameter_zero_changes_sparsity_and_forces_full_rebuild():
    m = Model("p5-sparsity")
    x = m.variable(lower=0.0, upper=10.0)
    a = m.parameter(value=1.0, name="a")
    m.add(a * x >= 0.0)
    m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        s.solve()
        a.value = 0.0
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.FULL_REBUILD
        assert r.trace.backend_reset is True
        assert r.trace.backend_reuse_applied is False
        assert r.trace.decision.execution_structure_changed


def test_indicator_certificate_revalidation_can_still_native_patch_after_safe_compile():
    m = Model("p5-indicator")
    x = m.variable(lower=0.0, upper=5.0, name="x")
    z = m.binary(name="z")
    p = m.parameter(value=1.0, name="p")
    m.indicator(z, p * x <= 2.0)
    m.add(z >= 1.0)
    m.maximize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        r0 = s.solve()
        cert0 = r0.compiled.precondition_certificate_hashes
        assert r0.objective == pytest.approx(2.0)
        p.value = 2.0
        r1 = s.solve()
        assert r1.trace.decision.certificate_changed
        assert r1.compiled.precondition_certificate_hashes != cert0
        assert r1.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r1.trace.backend_reuse_applied is True
        assert r1.trace.original_validation_valid
        assert r1.objective == pytest.approx(1.0)


def test_osqp_persistent_objective_parameter_uses_native_patch():
    m = Model("p5-qp")
    x = m.variable(lower=-5.0, upper=5.0, name="x")
    q = m.parameter(value=-2.0, name="q")
    m.minimize(x * x + q * x)
    backend, caps = _osqp_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        r0 = s.solve()
        assert r0.objective == pytest.approx(-1.0, abs=1e-7)
        q.value = -4.0
        r1 = s.solve()
        assert r1.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r1.trace.backend_reuse_applied is True
        assert r1.objective == pytest.approx(-4.0, abs=1e-7)
        assert r1.x[0] == pytest.approx(2.0, abs=1e-6)


def test_unverified_backend_falls_back_to_safe_rebuild_without_reuse_claim():
    m = Model("p5-fallback")
    x = m.variable(lower=0.0, upper=10.0)
    rhs = m.parameter(value=1.0)
    m.add(x >= rhs)
    m.minimize(x)
    backend = ScipyHighsBackend()
    caps = conform_persistent_backend(backend).manifest
    assert not caps.supports_verified(CapabilityKey.LIFECYCLE_PERSISTENT)
    with PersistentSession(m, backend, capabilities=caps) as s:
        assert s.solve().objective == pytest.approx(1.0)
        rhs.value = 2.0
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.SAFE_REBUILD
        assert r.trace.backend_reset
        assert r.objective == pytest.approx(2.0)


def test_semantic_structure_change_forces_full_rebuild():
    m = Model("p5-semantic-change")
    x = m.variable(lower=0.0, upper=10.0)
    m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        s.solve()
        m.add(x >= 3.0)
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.FULL_REBUILD
        assert r.trace.decision.semantic_changed
        assert r.trace.backend_reset
        assert r.objective == pytest.approx(3.0)


def test_snapshot_revisit_routes_from_current_backend_state_not_stale_snapshot_state():
    m = Model("p5-revisit")
    x = m.variable(lower=0.0, upper=10.0)
    rhs = m.parameter(value=1.0)
    m.add(x >= rhs)
    m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        assert s.solve().objective == pytest.approx(1.0)
        rhs.value = 3.0
        assert s.solve().objective == pytest.approx(3.0)
        rhs.value = 1.0
        r = s.solve()
        assert r.compiled.compilation_report.cache_status == "snapshot_hit"
        assert r.trace.path is MutationExecutionPath.NATIVE_PATCH
        assert r.trace.backend_reuse_applied is True
        assert r.objective == pytest.approx(1.0)


def test_repeated_no_mutation_is_not_misreported_as_mutation_patch():
    m = Model("p5-noop")
    x = m.variable(lower=0.0, upper=5.0)
    m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        s.solve()
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.NO_MUTATION_REUSE
        assert not r.trace.decision.mutation_kinds


def test_version_drifted_persistence_evidence_downgrades_to_safe_rebuild():
    from dataclasses import replace
    m = Model("p5-drift")
    x = m.variable(lower=0.0, upper=10.0)
    rhs = m.parameter(value=1.0)
    m.add(x >= rhs); m.minimize(x)
    backend, caps = _highs_binding()
    drifted = replace(caps, adapter_version="99.0.0")
    with PersistentSession(m, backend, capabilities=drifted) as s:
        s.solve(); rhs.value = 2.0
        r = s.solve()
        assert r.trace.path is MutationExecutionPath.SAFE_REBUILD
        assert CapabilityKey.LIFECYCLE_PERSISTENT in r.trace.decision.missing_capabilities


def test_p5_trace_json_schema_validates_serialized_trace():
    import json, importlib.resources, jsonschema
    m = Model("p5-schema")
    x = m.variable(lower=0.0, upper=2.0); m.minimize(x)
    backend, caps = _highs_binding()
    with PersistentSession(m, backend, capabilities=caps) as s:
        trace = s.solve().trace.to_dict()
    schema = json.loads(importlib.resources.files("solverpilot.session").joinpath("persistent-session-trace.schema.json").read_text())
    jsonschema.validate(trace, schema)


def test_capability_manifest_identity_mismatch_is_rejected():
    from dataclasses import replace
    backend, caps = _highs_binding()
    bad = replace(caps, backend="not-the-backend")
    m = Model(); x = m.variable(lower=0.0, upper=1.0); m.minimize(x)
    with pytest.raises(Exception, match="identity"):
        PersistentSession(m, backend, capabilities=bad)
    backend.close()
