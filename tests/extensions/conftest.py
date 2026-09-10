from __future__ import annotations

from dataclasses import dataclass
from threading import Event

import numpy as np

from solverpilot.backends.base import BackendSolveResult
from solverpilot.bridges import BridgeSpec, ExactnessClass
from solverpilot.capabilities import BackendManifest
from solverpilot.extensions import (
    ExtensionContributions,
    ExtensionKind,
    ExtensionManifest,
    ExtensionSelfCheck,
    NamedExtensionExport,
)


@dataclass
class FakeBackend:
    name: str
    version: str = "1.0.0"

    def __post_init__(self):
        self._manifest = BackendManifest(name=self.name, version=self.version)

    @property
    def manifest(self):
        return self._manifest

    def is_available(self) -> bool:
        return True

    def solve(self, problem):
        return BackendSolveResult("ok", np.zeros(problem.n_variables), 0.0, {})


class FakeExtension:
    def __init__(
        self,
        extension_id: str,
        *,
        version: str = "1.0.0",
        dependencies=(),
        backends=(),
        bridges=(),
        exports=(),
        entrypoint=None,
        kind=ExtensionKind.MIXED,
    ) -> None:
        self._manifest = ExtensionManifest(
            extension_id=extension_id,
            name=extension_id,
            version=version,
            kind=kind,
            description="test extension",
            dependencies=dependencies,
            entrypoint=entrypoint,
        )
        self._backends = tuple(backends)
        self._bridges = tuple(bridges)
        self._exports = tuple(exports)
        self.check_ok = True
        self.raise_check = None
        self.raise_contrib = None
        self.contrib_entered: Event | None = None
        self.contrib_release: Event | None = None

    @property
    def manifest(self):
        return self._manifest

    def self_check(self):
        if self.raise_check is not None:
            raise self.raise_check
        return ExtensionSelfCheck(ok=self.check_ok, checks=("ok",))

    def contributions(self):
        if self.contrib_entered is not None:
            self.contrib_entered.set()
        if self.contrib_release is not None:
            self.contrib_release.wait(timeout=10)
        if self.raise_contrib is not None:
            raise self.raise_contrib
        return ExtensionContributions(
            backends=self._backends,
            bridges=self._bridges,
            exports=self._exports,
        )


def bridge(bridge_id: str = "test.bridge", version: str = "1.0.0") -> BridgeSpec:
    return BridgeSpec(
        bridge_id=bridge_id,
        bridge_version=version,
        source_kind="indicator",
        exactness=ExactnessClass.EXACT_EQUIVALENT,
        priority=50,
        description="test bridge",
    )


def export(name: str = "demo", value=None) -> NamedExtensionExport:
    if value is None:
        value = object()
    return NamedExtensionExport("reporter", name, value)
