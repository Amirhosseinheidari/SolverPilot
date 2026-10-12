"""Routing bindings hash public configuration without copying native state."""

from dataclasses import dataclass, field, replace
from threading import RLock

import pytest
import numpy as np

from solverpilot.backends import HighspyNativeBackend
from solverpilot.experimental.lp_session import LPRoutingSession, _backend_snapshot


def bound_to(backends):
    session = object.__new__(LPRoutingSession)
    object.__setattr__(session, "_backend_settings", _backend_snapshot(backends))
    return session


def test_highspy_native_state_does_not_invalidate_bound_settings():
    backend = HighspyNativeBackend(reuse=True, solver="simplex", threads=1)
    backends = {"native": backend}
    session = bound_to(backends)
    # These objects cannot be deep-copied/serialized but are not configuration.
    backend._highs = object()
    backend._matrix_hash = "a different last model"
    assert session.backends_match(backends)
    backend.reset()
    assert session.backends_match(backends)
    assert not session.backends_match({"native": replace(backend)})
    backend.presolve = False
    assert not session.backends_match(backends)
    backend.presolve = None
    assert session.backends_match(backends)
    backend.reuse = False
    assert not session.backends_match(backends)


@dataclass
class OpaqueSettingsBackend:
    setting: object = "serializable"
    runtime: object = field(default_factory=RLock, init=False)
    _private: object = field(default_factory=RLock)


@pytest.mark.parametrize("opaque", [object(), RLock(), {"callback": lambda: None}])
def test_opaque_public_settings_fail_closed_instead_of_hashing_runtime_labels(opaque):
    backend = OpaqueSettingsBackend()
    session = bound_to({"a": backend})
    backend.setting = opaque
    with pytest.raises(TypeError, match="not JSON-serializable"):
        _backend_snapshot({"a": backend})
    assert not session.backends_match({"a": backend})


def test_serializable_nested_settings_and_backend_identity_are_bound():
    backend = OpaqueSettingsBackend(setting={"options": [1, 2]})
    backends = {"a": backend}
    session = bound_to(backends)
    backend.runtime = object()
    backend._private = object()
    assert session.backends_match(backends)
    backend.setting["options"][0] = 3
    assert not session.backends_match(backends)
    with pytest.raises(ValueError, match="dataclass"):
        _backend_snapshot({"opaque": object()})


@pytest.mark.parametrize("setting", [float('inf'), float('-inf'), float('nan'),
    {"nested": [np.float64('inf')]}, np.array([1., float('nan')])])
def test_nonfinite_settings_cannot_collide_with_text(setting):
    backend = OpaqueSettingsBackend(setting="inf")
    session = bound_to({"a": backend})
    backend.setting = setting
    with pytest.raises(ValueError, match="finite"):
        _backend_snapshot({"a": backend})
    assert not session.backends_match({"a": backend})


def test_nonstring_mapping_keys_cannot_collide_with_string_keys():
    backend = OpaqueSettingsBackend(setting={"1": "value"})
    session = bound_to({"a": backend})
    backend.setting = {1: "value"}
    assert not session.backends_match({"a": backend})
