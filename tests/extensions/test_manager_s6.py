from __future__ import annotations

from dataclasses import replace

import pytest

from solverpilot.backends import BackendRegistry
from solverpilot.bridges import BridgeRegistry
from solverpilot.extensions import (
    ExtensionActivationError,
    ExtensionContributions,
    ExtensionDependencyError,
    ExtensionManager,
    ExtensionRegistrationError,
    ExtensionRegistryFrozenError,
    ExtensionValidationError,
    ExportKind,
    NamedExtensionExport,
)

from .conftest import FakeBackend, FakeExtension, bridge, export


def registries():
    return BackendRegistry(), BridgeRegistry()


def test_registration_and_deterministic_listing():
    manager = ExtensionManager()
    manager.register(FakeExtension("z.demo"))
    manager.register(FakeExtension("a.demo"))
    assert manager.list_extension_ids() == ("a.demo", "z.demo")
    assert manager.get("Z.DEMO").manifest.extension_id == "z.demo"


def test_duplicate_registration_is_rejected():
    manager = ExtensionManager([FakeExtension("a.demo")])
    with pytest.raises(ExtensionRegistrationError):
        manager.register(FakeExtension("a.demo"))


def test_batch_registration_is_duplicate_safe():
    manager = ExtensionManager()
    with pytest.raises(ExtensionRegistrationError):
        manager.register_many((FakeExtension("a.demo"), FakeExtension("a.demo")))
    assert manager.list_extension_ids() == ()


def test_failed_self_check_and_exception_do_not_register():
    manager = ExtensionManager()
    bad = FakeExtension("bad.demo")
    bad.check_ok = False
    with pytest.raises(ExtensionValidationError):
        manager.register(bad)
    boom = FakeExtension("boom.demo")
    boom.raise_check = RuntimeError("boom")
    with pytest.raises(ExtensionValidationError):
        manager.register(boom)
    assert manager.list_extension_ids() == ()


def test_missing_dependency_blocks_freeze():
    manager = ExtensionManager([FakeExtension("a.demo", dependencies=("missing.demo>=1.0.0",))])
    with pytest.raises(ExtensionDependencyError):
        manager.freeze()


def test_version_mismatch_blocks_freeze():
    manager = ExtensionManager(
        [
            FakeExtension("core.demo", version="1.0.0"),
            FakeExtension("consumer.demo", dependencies=("core.demo>=2.0.0",)),
        ]
    )
    with pytest.raises(ExtensionDependencyError):
        manager.freeze()


def test_dependency_cycle_blocks_freeze():
    manager = ExtensionManager(
        [
            FakeExtension("a.demo", dependencies=("b.demo",)),
            FakeExtension("b.demo", dependencies=("a.demo",)),
        ]
    )
    with pytest.raises(ExtensionDependencyError):
        manager.freeze()


def test_dependency_order_is_topological_and_deterministic():
    manager = ExtensionManager(
        [
            FakeExtension("z.demo", dependencies=("core.demo",)),
            FakeExtension("a.demo", dependencies=("core.demo",)),
            FakeExtension("core.demo"),
        ]
    )
    assert manager.dependency_order() == ("core.demo", "a.demo", "z.demo")
    manager.freeze()
    assert manager.frozen is True


def test_frozen_manager_rejects_registration():
    manager = ExtensionManager([FakeExtension("a.demo")])
    manager.freeze()
    with pytest.raises(ExtensionRegistryFrozenError):
        manager.register(FakeExtension("b.demo"))


def test_activation_requires_freeze():
    manager = ExtensionManager([FakeExtension("a.demo")])
    backends, bridges = registries()
    with pytest.raises(ExtensionActivationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)


def test_contribution_exception_does_not_touch_canonical_registries():
    extension = FakeExtension("a.demo", backends=(FakeBackend("new-backend"),))
    extension.raise_contrib = RuntimeError("boom")
    manager = ExtensionManager([extension])
    manager.freeze()
    backends, bridges = registries()
    with pytest.raises(ExtensionValidationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ()
    assert bridges.specs == ()


def test_backend_and_bridge_activation_use_canonical_registries():
    backend = FakeBackend("ext-backend")
    spec = bridge("ext.bridge")
    manager = ExtensionManager([FakeExtension("a.demo", backends=(backend,), bridges=(spec,))])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ()
    assert bridges.specs == ()
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.get("ext-backend") is backend
    assert any(item.bridge_id == "ext.bridge" for item in bridges.specs)


def test_named_exports_are_read_only_catalog_entries_after_activation():
    value = object()
    manager = ExtensionManager([FakeExtension("a.demo", exports=(export("demo", value),))])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert manager.get_export(ExportKind.REPORTER, "demo") is value
    assert manager.list_exports(kind="reporter")[0].name == "demo"


def test_target_backend_collision_is_rejected_before_mutation():
    manager = ExtensionManager([FakeExtension("a.demo", backends=(FakeBackend("same"),))])
    manager.freeze()
    backends, bridges = registries()
    existing = FakeBackend("same")
    backends.register(existing)
    with pytest.raises(ExtensionActivationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    assert backends.get("same") is existing


def test_target_bridge_collision_is_rejected_before_mutation():
    spec = bridge("same.bridge")
    manager = ExtensionManager([FakeExtension("a.demo", bridges=(spec,))])
    manager.freeze()
    backends, bridges = registries()
    bridges.register(spec)
    with pytest.raises(ExtensionActivationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    assert len(bridges.specs) == 1


def test_duplicate_staged_backend_across_extensions_is_rejected():
    manager = ExtensionManager(
        [
            FakeExtension("a.demo", backends=(FakeBackend("same"),)),
            FakeExtension("b.demo", backends=(FakeBackend("same"),)),
        ]
    )
    manager.freeze()
    backends, bridges = registries()
    with pytest.raises(ExtensionActivationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)


def test_plan_is_single_use():
    manager = ExtensionManager([FakeExtension("a.demo")])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    with pytest.raises(ExtensionActivationError):
        manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)


def test_forged_plan_object_is_rejected():
    manager = ExtensionManager([FakeExtension("a.demo")])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    forged = replace(plan)
    with pytest.raises(ExtensionActivationError):
        manager.apply_activation(forged, backend_registry=backends, bridge_registry=bridges)


def test_mutated_backend_after_staging_invalidates_plan():
    backend = FakeBackend("before")
    manager = ExtensionManager([FakeExtension("a.demo", backends=(backend,))])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    backend._manifest = backend._manifest.__class__(name="after", version="1.0.0")
    with pytest.raises(ExtensionActivationError):
        manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ()


def test_target_registry_drift_after_staging_invalidates_plan():
    manager = ExtensionManager([FakeExtension("a.demo", backends=(FakeBackend("new"),))])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    backends.register(FakeBackend("external"))
    with pytest.raises(ExtensionActivationError):
        manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ("external",)


def test_manifest_drift_after_registration_is_detected():
    extension = FakeExtension("a.demo")
    manager = ExtensionManager([extension])
    extension._manifest = replace(extension._manifest, description="changed")
    with pytest.raises(ExtensionValidationError):
        manager.freeze()


def test_self_check_is_repeated_before_staging():
    extension = FakeExtension("a.demo")
    manager = ExtensionManager([extension])
    manager.freeze()
    extension.check_ok = False
    backends, bridges = registries()
    with pytest.raises(ExtensionValidationError):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)


def test_backend_manifest_capability_or_metadata_drift_invalidates_plan():
    from solverpilot.capabilities import Capability, SupportLevel

    backend = FakeBackend("drift-backend")
    backend._manifest.capabilities[Capability.LP] = SupportLevel.NATIVE
    backend._manifest.metadata["source"] = {"revision": 1}
    manager = ExtensionManager([FakeExtension("drift.demo", backends=(backend,))])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    backend._manifest.metadata["source"]["revision"] = 2
    with pytest.raises(ExtensionActivationError):
        manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ()


def test_activation_plan_is_bound_to_exact_registry_objects():
    manager = ExtensionManager([FakeExtension("bound.demo")])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    other_backends, other_bridges = registries()
    with pytest.raises(ExtensionActivationError, match="different canonical registry objects"):
        manager.apply_activation(
            plan, backend_registry=other_backends, bridge_registry=other_bridges
        )
    # Rejection does not consume the correctly-bound plan.
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)


def test_backend_nonfinite_or_cyclic_observable_state_fails_closed():
    import math
    from solverpilot.extensions import ExtensionValidationError

    backend = FakeBackend("bad-meta")
    backend._manifest.metadata["bad"] = math.nan
    manager = ExtensionManager([FakeExtension("meta.demo", backends=(backend,))])
    manager.freeze()
    backends, bridges = registries()
    with pytest.raises(ExtensionValidationError, match="NaN or Infinity"):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)

    backend2 = FakeBackend("cyclic-meta")
    cyclic = {}
    cyclic["self"] = cyclic
    backend2._manifest.metadata["bad"] = cyclic
    manager2 = ExtensionManager([FakeExtension("cycle.demo", backends=(backend2,))])
    manager2.freeze()
    backends2, bridges2 = registries()
    with pytest.raises(ExtensionValidationError, match="cyclic"):
        manager2.plan_activation(backend_registry=backends2, bridge_registry=bridges2)


def test_non_mixed_manifest_kind_rejects_mismatched_contribution():
    manager = ExtensionManager([
        FakeExtension(
            "reporter.kind",
            kind=__import__("solverpilot.extensions", fromlist=["ExtensionKind"]).ExtensionKind.REPORTER,
            backends=(FakeBackend("wrong-kind"),),
        )
    ])
    manager.freeze()
    backends, bridges = registries()
    with pytest.raises(ExtensionValidationError, match="declares kind"):
        manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ()


def test_mixed_manifest_kind_accepts_multiple_contribution_kinds():
    value = object()
    manager = ExtensionManager([
        FakeExtension(
            "mixed.kind",
            backends=(FakeBackend("mixed-backend"),),
            exports=(export("mixed-report", value),),
        )
    ])
    manager.freeze()
    backends, bridges = registries()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.get("mixed-backend").manifest.name == "mixed-backend"
    assert manager.get_export("reporter", "mixed-report") is value
