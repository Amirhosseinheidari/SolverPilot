from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from threading import Event

import pytest

import solverpilot
from solverpilot.extensions import ExtensionManager, ExtensionRegistrationError

from .conftest import FakeExtension


def test_concurrent_unique_registration_preserves_all_extensions():
    manager = ExtensionManager()
    ids = [f"ext.demo{i:02d}" for i in range(20)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(lambda key: manager.register(FakeExtension(key)), ids))
    assert manager.list_extension_ids() == tuple(sorted(ids))


def test_concurrent_duplicate_registration_has_exactly_one_winner():
    manager = ExtensionManager()

    def attempt(_):
        try:
            manager.register(FakeExtension("race.demo"))
        except ExtensionRegistrationError:
            return False
        return True

    with ThreadPoolExecutor(max_workers=12) as pool:
        outcomes = list(pool.map(attempt, range(12)))
    assert sum(outcomes) == 1
    assert manager.list_extension_ids() == ("race.demo",)


def test_contributions_hook_does_not_hold_manager_lock():
    entered = Event()
    release = Event()
    extension = FakeExtension("blocking.demo")
    extension.contrib_entered = entered
    extension.contrib_release = release
    manager = ExtensionManager([extension])
    manager.freeze()

    from solverpilot.backends import BackendRegistry
    from solverpilot.bridges import BridgeRegistry

    with ThreadPoolExecutor(max_workers=2) as pool:
        future = pool.submit(
            manager.plan_activation,
            backend_registry=BackendRegistry(),
            bridge_registry=BridgeRegistry(),
        )
        assert entered.wait(timeout=3)
        # If contributions() held the manager lock, this future would block until release.
        query = pool.submit(manager.list_extension_ids)
        assert query.result(timeout=2) == ("blocking.demo",)
        release.set()
        future.result(timeout=5)


def test_top_level_public_api_remains_frozen():
    assert len(solverpilot.__all__) == 78
    assert "ExtensionManager" not in solverpilot.__all__


def test_extensions_are_available_only_from_subpackage_surface():
    from solverpilot.extensions import ExtensionManager as Manager

    assert Manager is ExtensionManager
    assert not hasattr(solverpilot, "ExtensionManager")
