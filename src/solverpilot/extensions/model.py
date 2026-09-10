from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import re
from typing import Any, Protocol, runtime_checkable

from solverpilot.backends.base import Backend
from solverpilot.bridges.registry import BridgeSpec

from .errors import ExtensionManifestError
from .manifest import ExtensionManifest

_NAME_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")


@dataclass(frozen=True, slots=True)
class ExtensionSelfCheck:
    ok: bool
    checks: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.ok, bool):
            raise ExtensionManifestError("self-check ok must be a boolean")
        checks = tuple(str(item).strip() for item in self.checks)
        warnings = tuple(str(item).strip() for item in self.warnings)
        if any(not item for item in checks):
            raise ExtensionManifestError("self-check identifiers must be non-empty")
        if any(not item for item in warnings):
            raise ExtensionManifestError("self-check warnings must be non-empty")
        object.__setattr__(self, "checks", checks)
        object.__setattr__(self, "warnings", warnings)


class ExportKind(str, Enum):
    DATA_CONNECTOR = "data_connector"
    FEATURE_EXTRACTOR = "feature_extractor"
    EVALUATOR = "evaluator"
    REPORTER = "reporter"
    APPLICATION = "application"


@dataclass(frozen=True, slots=True)
class NamedExtensionExport:
    kind: ExportKind | str
    name: str
    value: Any

    def __post_init__(self) -> None:
        try:
            kind = self.kind if isinstance(self.kind, ExportKind) else ExportKind(str(self.kind).strip().lower())
        except ValueError as exc:
            raise ExtensionManifestError(f"unsupported export kind {self.kind!r}") from exc
        name = str(self.name).strip().lower()
        if not _NAME_RE.fullmatch(name):
            raise ExtensionManifestError(f"invalid export name {self.name!r}")
        if self.value is None:
            raise ExtensionManifestError("export value may not be None")
        object.__setattr__(self, "kind", kind)
        object.__setattr__(self, "name", name)

    @property
    def key(self) -> str:
        return f"{self.kind.value}:{self.name}"


@dataclass(frozen=True, slots=True)
class ExtensionContributions:
    backends: tuple[Backend, ...] = ()
    bridges: tuple[BridgeSpec, ...] = ()
    exports: tuple[NamedExtensionExport, ...] = ()


@runtime_checkable
class Extension(Protocol):
    @property
    def manifest(self) -> ExtensionManifest: ...

    def self_check(self) -> ExtensionSelfCheck: ...

    def contributions(self) -> ExtensionContributions: ...


@dataclass(frozen=True, slots=True)
class StagedExtensionContribution:
    extension_id: str
    extension_version: str
    backends: tuple[Backend, ...] = ()
    bridges: tuple[BridgeSpec, ...] = ()
    exports: tuple[NamedExtensionExport, ...] = ()


@dataclass(frozen=True, slots=True)
class ExtensionActivationPlan:
    generation: int
    extension_order: tuple[str, ...]
    staged: tuple[StagedExtensionContribution, ...]
    target_backend_names: tuple[str, ...]
    target_bridge_keys: tuple[str, ...]
    plan_sha256: str
