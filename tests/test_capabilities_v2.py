from __future__ import annotations

from dataclasses import dataclass
import json

import numpy as np
import pytest

from solverpilot.backends import BackendSolveResult, ScipyHighsBackend, ScipyHighsLPBackend, ScipySLSQPQPBackend
from solverpilot.capabilities import (
    BackendCapabilityManifestV2,
    BackendManifest,
    Capability,
    CapabilityClaim,
    CapabilityEvidence,
    CapabilityKey,
    CapabilityMode,
    CapabilityStatus,
    SupportLevel,
    VerificationLevel,
    conform_backend_problem_classes,
    project_legacy_manifest,
    with_verified_claims,
)


def evidence(*, backend="1.0", binding="2.0", adapter="0.0.34"):
    return CapabilityEvidence(
        evidence_id="p3:test",
        kind="contract_test",
        backend_versions=(backend,),
        binding_versions=(binding,),
        adapter_versions=(adapter,),
        verified_on="2026-09-06",
    )


def manifest_with(claim: CapabilityClaim, *, backend_version="1.0", binding_version="2.0"):
    return BackendCapabilityManifestV2(
        backend="fake",
        backend_version=backend_version,
        binding="fake-binding",
        binding_version=binding_version,
        adapter_version="0.0.34",
        claims={CapabilityKey.PROBLEM_LP: claim},
    )


def test_verified_native_capability_is_usable():
    claim = CapabilityClaim(
        CapabilityStatus.SUPPORTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        evidence=(evidence(),),
    )
    m = manifest_with(claim)
    assert m.supports_verified(CapabilityKey.PROBLEM_LP)


def test_version_drift_invalidates_evidence_fail_closed():
    claim = CapabilityClaim(
        CapabilityStatus.SUPPORTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        evidence=(evidence(backend="1.0"),),
    )
    m = manifest_with(claim, backend_version="1.1")
    check = m.check(CapabilityKey.PROBLEM_LP)
    assert not check.usable
    assert "runtime version" in check.reason


def test_safe_emulation_requires_explicit_policy():
    claim = CapabilityClaim(
        CapabilityStatus.RESTRICTED,
        CapabilityMode.EMULATED_SAFE,
        VerificationLevel.VERIFIED,
        restrictions={"algorithm": "adapter-emulation"},
        evidence=(evidence(),),
    )
    m = manifest_with(claim)
    assert not m.supports_verified(CapabilityKey.PROBLEM_LP)
    assert m.supports_verified(CapabilityKey.PROBLEM_LP, allow_safe_emulation=True)


def test_risky_emulation_is_disabled_even_when_verified_unless_explicit():
    claim = CapabilityClaim(
        CapabilityStatus.RESTRICTED,
        CapabilityMode.EMULATED_RISKY,
        VerificationLevel.VERIFIED,
        restrictions={"risk": "semantics may differ"},
        evidence=(evidence(),),
    )
    m = manifest_with(claim)
    assert not m.supports_verified(CapabilityKey.PROBLEM_LP, allow_safe_emulation=True)
    assert m.supports_verified(CapabilityKey.PROBLEM_LP, allow_risky_emulation=True)


def test_restricted_claim_requires_machine_readable_restrictions():
    with pytest.raises(ValueError):
        CapabilityClaim(
            CapabilityStatus.RESTRICTED,
            CapabilityMode.NATIVE,
            VerificationLevel.PROJECTED_LEGACY,
        )


def test_legacy_native_projection_does_not_upgrade_to_verified_v2():
    old = BackendManifest("legacy", version="1", capabilities={Capability.LP: SupportLevel.NATIVE})
    m = project_legacy_manifest(old, binding="x", binding_version="2", adapter_version="0.0.34")
    claim = m.claim(CapabilityKey.PROBLEM_LP)
    assert claim.status is CapabilityStatus.SUPPORTED
    assert claim.mode is CapabilityMode.NATIVE
    assert claim.verification is VerificationLevel.PROJECTED_LEGACY
    assert not m.supports_verified(CapabilityKey.PROBLEM_LP)


def test_legacy_safe_emulation_is_preserved_as_restricted():
    old = BackendManifest("legacy", version="1", capabilities={Capability.CONVEX_QP: SupportLevel.EMULATED_SAFE})
    m = project_legacy_manifest(old, adapter_version="0.0.34")
    claim = m.claim(CapabilityKey.PROBLEM_CONVEX_QP)
    assert claim.status is CapabilityStatus.RESTRICTED
    assert claim.mode is CapabilityMode.EMULATED_SAFE
    assert claim.restrictions["legacy_emulation"] == "emulated_safe"


def test_coarse_same_sparsity_projection_is_not_granular_verified_support():
    old = BackendManifest(
        "legacy",
        version="1",
        capabilities={Capability.SAME_SPARSITY_DATA_UPDATE: SupportLevel.NATIVE},
    )
    m = project_legacy_manifest(old, adapter_version="0.0.34")
    for key in (
        CapabilityKey.INCREMENTAL_VARIABLE_BOUNDS,
        CapabilityKey.INCREMENTAL_RHS,
        CapabilityKey.INCREMENTAL_OBJECTIVE,
        CapabilityKey.INCREMENTAL_MATRIX_VALUES,
    ):
        claim = m.claim(key)
        assert claim.status is CapabilityStatus.RESTRICTED
        assert claim.verification is VerificationLevel.PROJECTED_LEGACY
        assert claim.restrictions["granular_operation_not_individually_verified"] is True
        assert not m.supports_verified(key)


def test_signature_is_canonical_and_changes_on_version_or_restriction():
    c = CapabilityClaim(
        CapabilityStatus.RESTRICTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        restrictions={"max_rows": 100, "types": ["linear"]},
        evidence=(evidence(),),
    )
    a = manifest_with(c)
    b = BackendCapabilityManifestV2(
        backend="fake",
        backend_version="1.0",
        binding="fake-binding",
        binding_version="2.0",
        adapter_version="0.0.34",
        claims={CapabilityKey.PROBLEM_LP: CapabilityClaim(
            CapabilityStatus.RESTRICTED,
            CapabilityMode.NATIVE,
            VerificationLevel.VERIFIED,
            restrictions={"types": ["linear"], "max_rows": 100},
            evidence=(evidence(),),
        )},
    )
    assert a.signature == b.signature
    changed_version = manifest_with(c, backend_version="1.1")
    assert a.signature != changed_version.signature
    changed_restriction = manifest_with(CapabilityClaim(
        CapabilityStatus.RESTRICTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        restrictions={"max_rows": 101, "types": ["linear"]},
        evidence=(evidence(),),
    ))
    assert a.signature != changed_restriction.signature


def test_p0_schema_projection_preserves_native_vs_unverified_as_supported_vs_restricted():
    verified = CapabilityClaim(
        CapabilityStatus.SUPPORTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        evidence=(evidence(),),
    )
    unverified = CapabilityClaim(
        CapabilityStatus.SUPPORTED,
        CapabilityMode.NATIVE,
        VerificationLevel.PROJECTED_LEGACY,
    )
    m = BackendCapabilityManifestV2(
        backend="fake", backend_version="1.0", binding="b", binding_version="2.0", adapter_version="0.0.34",
        claims={CapabilityKey.PROBLEM_LP: verified, CapabilityKey.PROBLEM_MILP: unverified},
    )
    d = m.to_p0_schema_dict()
    assert d["schema_version"] == "2.0"
    assert d["problem_classes"]["lp"] == "supported"
    assert d["problem_classes"]["milp"] == "restricted"
    assert d["restrictions"]["problem.milp"]["verification"] == "projected_legacy"


def test_actual_scipy_highs_lp_conformance_verifies_native_lp_and_result_channels():
    report = conform_backend_problem_classes(ScipyHighsLPBackend(method="highs-ds"))
    assert report.available
    assert report.passed
    assert report.manifest.supports_verified(CapabilityKey.PROBLEM_LP)
    assert report.manifest.supports_verified(CapabilityKey.CONSTRAINT_LINEAR)
    assert report.manifest.supports_verified(CapabilityKey.RESULT_PRIMAL)
    assert report.manifest.supports_verified(CapabilityKey.RESULT_OBJECTIVE)


def test_actual_scipy_highs_bridge_conformance_verifies_lp_and_milp():
    report = conform_backend_problem_classes(ScipyHighsBackend())
    assert report.passed
    assert report.manifest.supports_verified(CapabilityKey.PROBLEM_LP)
    assert report.manifest.supports_verified(CapabilityKey.PROBLEM_MILP)


def test_emulated_qp_can_be_conformance_verified_but_not_used_as_native_by_default():
    report = conform_backend_problem_classes(ScipySLSQPQPBackend())
    claim = report.manifest.claim(CapabilityKey.PROBLEM_CONVEX_QP)
    assert claim.verification is VerificationLevel.VERIFIED
    assert claim.mode is CapabilityMode.EMULATED_SAFE
    assert not report.manifest.supports_verified(CapabilityKey.PROBLEM_CONVEX_QP)
    assert report.manifest.supports_verified(CapabilityKey.PROBLEM_CONVEX_QP, allow_safe_emulation=True)


@dataclass(slots=True)
class InvalidClaimingLPBackend:
    @property
    def manifest(self):
        return BackendManifest("invalid-v2", version="1", capabilities={Capability.LP: SupportLevel.NATIVE})

    def is_available(self):
        return True

    def solve(self, problem):
        return BackendSolveResult("optimal", np.array([0.0]), 0.0, {})


def test_invalid_native_smoke_does_not_upgrade_claim_to_verified():
    report = conform_backend_problem_classes(InvalidClaimingLPBackend())
    assert not report.passed
    assert not report.manifest.supports_verified(CapabilityKey.PROBLEM_LP)
    assert report.manifest.claim(CapabilityKey.PROBLEM_LP).verification is VerificationLevel.PROJECTED_LEGACY


def test_with_verified_claims_binds_evidence_to_exact_runtime_identity():
    projected = BackendCapabilityManifestV2(
        backend="x", backend_version="1.0", binding="b", binding_version="2.0", adapter_version="0.0.34"
    )
    verified = with_verified_claims(projected, {CapabilityKey.PROBLEM_LP: evidence()})
    assert verified.supports_verified(CapabilityKey.PROBLEM_LP)
    drifted = BackendCapabilityManifestV2(
        backend="x", backend_version="1.0", binding="b", binding_version="2.1", adapter_version="0.0.34",
        claims=verified.claims,
    )
    assert not drifted.supports_verified(CapabilityKey.PROBLEM_LP)


def test_canonical_manifest_roundtrips_through_json():
    old = BackendManifest("legacy", version="1", capabilities={Capability.LP: SupportLevel.NATIVE})
    m = project_legacy_manifest(old, adapter_version="0.0.34")
    payload = json.loads(json.dumps(m.to_canonical_dict(), sort_keys=True))
    assert payload["schema_version"] == "2.0"
    assert payload["claims"]["problem.lp"]["mode"] == "native"


def test_registry_resolves_v2_manifest_without_breaking_v1_registry():
    from solverpilot.backends import BackendRegistry
    r = BackendRegistry()
    r.register(ScipyHighsLPBackend(method="highs-ds"))
    projected = r.capability_manifest_v2("scipy-highs-ds", verify=False)
    assert projected.claim(CapabilityKey.PROBLEM_LP).verification is VerificationLevel.PROJECTED_LEGACY
    verified = r.capability_manifest_v2("scipy-highs-ds", verify=True)
    assert verified.supports_verified(CapabilityKey.PROBLEM_LP)


def test_full_builtin_inventory_can_be_resolved_and_verified_without_crashing():
    from solverpilot.runtime import builtin_backend_candidates
    from solverpilot.capabilities import conform_backends
    backends = builtin_backend_candidates()
    reports = conform_backends(backends)
    assert len(reports) == len(backends)
    assert {r.backend for r in reports} == {b.manifest.name for b in backends}
    for report in reports:
        assert len(report.manifest.signature) == 64
        if not report.available:
            assert not report.passed


def test_bundled_advanced_legacy_claims_are_not_upgraded_without_granular_conformance():
    from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
    for backend, keys in [
        (BundledHighsCAPIBackend(), (CapabilityKey.START_BASIS, CapabilityKey.INCREMENTAL_RHS)),
        (BundledOSQPCAPIBackend(), (CapabilityKey.START_PRIMAL, CapabilityKey.START_DUAL, CapabilityKey.INCREMENTAL_MATRIX_VALUES)),
    ]:
        report = conform_backend_problem_classes(backend)
        assert report.available
        assert report.passed
        for key in keys:
            assert report.manifest.claim(key).verification is not VerificationLevel.VERIFIED
            assert not report.manifest.supports_verified(key)


def test_v2_problem_requirements_and_compatibility_use_verified_native_surface():
    from solverpilot import LinearProblem, VariableDomain
    from solverpilot.capabilities import compatible_v2, requirements_v2_for
    lp = LinearProblem.from_data(
        A=[[1.0]], c=[1.0], variable_lower=[0.0], variable_upper=[2.0],
        constraint_lower=[1.0], constraint_upper=[np.inf], domains=[VariableDomain.CONTINUOUS],
    )
    report = conform_backend_problem_classes(ScipyHighsLPBackend(method="highs-ds"))
    req = requirements_v2_for(lp)
    ok, checks = compatible_v2(report.manifest, req)
    assert ok
    assert all(c.usable for c in checks)
    projected = project_legacy_manifest(ScipyHighsLPBackend(method="highs-ds").manifest, adapter_version="0.0.34")
    ok2, checks2 = compatible_v2(projected, req)
    assert not ok2
    assert any(not c.usable for c in checks2)


def test_capability_schema_resource_is_available():
    import importlib.resources as resources
    text = resources.files("solverpilot.capabilities").joinpath("backend-capability-v2.schema.json").read_text()
    payload = json.loads(text)
    assert payload["$id"] == "solverpilot.backend-capability-v2"
    assert payload["properties"]["schema_version"]["const"] == "2.0"


def test_explicit_v2_provider_is_resolved_without_legacy_reinterpretation():
    from solverpilot.capabilities import resolve_backend_capabilities

    class ExplicitBackend:
        @property
        def manifest(self):
            return BackendManifest("explicit", version="legacy", capabilities={Capability.LP: SupportLevel.UNSUPPORTED})

        @property
        def capability_manifest_v2(self):
            return manifest_with(CapabilityClaim(
                CapabilityStatus.SUPPORTED,
                CapabilityMode.NATIVE,
                VerificationLevel.VERIFIED,
                evidence=(evidence(),),
            ))

    resolved = resolve_backend_capabilities(ExplicitBackend())
    assert resolved.supports_verified(CapabilityKey.PROBLEM_LP)


def test_manifest_and_restrictions_are_deeply_immutable_for_signature_stability():
    claim = CapabilityClaim(
        CapabilityStatus.RESTRICTED,
        CapabilityMode.NATIVE,
        VerificationLevel.VERIFIED,
        restrictions={"nested": {"limit": 2}, "kinds": ["a", "b"]},
        evidence=(evidence(),),
    )
    m = manifest_with(claim)
    sig = m.signature
    with pytest.raises(TypeError):
        m.claims[CapabilityKey.PROBLEM_MILP] = claim
    with pytest.raises(TypeError):
        claim.restrictions["x"] = 1
    with pytest.raises(TypeError):
        claim.restrictions["nested"]["limit"] = 3
    assert m.signature == sig
