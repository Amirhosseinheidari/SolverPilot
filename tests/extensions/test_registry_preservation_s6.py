from __future__ import annotations

from solverpilot.backends import BackendRegistry
from solverpilot.bridges import BridgePolicy, BridgeRegistry, default_bridge_registry
from solverpilot.extensions import ExtensionManager

from .conftest import FakeBackend, FakeExtension


def test_empty_extension_activation_leaves_backend_registry_semantics_unchanged():
    existing = FakeBackend("existing")
    backends = BackendRegistry()
    backends.register(existing)
    bridges = BridgeRegistry()
    manager = ExtensionManager([FakeExtension("empty.demo")])
    manager.freeze()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    assert backends.names() == ("existing",)
    assert backends.get("existing") is existing
    assert backends.available() == (existing,)


def test_empty_extension_activation_leaves_default_bridge_order_unchanged():
    bridges = default_bridge_registry()
    before = bridges.candidates("indicator", policy=BridgePolicy())
    manager = ExtensionManager([FakeExtension("empty.demo")])
    manager.freeze()
    backends = BackendRegistry()
    plan = manager.plan_activation(backend_registry=backends, bridge_registry=bridges)
    manager.apply_activation(plan, backend_registry=backends, bridge_registry=bridges)
    after = bridges.candidates("indicator", policy=BridgePolicy())
    assert after == before
