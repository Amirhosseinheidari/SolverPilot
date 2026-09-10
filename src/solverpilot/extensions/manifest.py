from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
import hashlib
import json
import math
import re
from types import MappingProxyType
from typing import Any, Mapping

from .errors import ExtensionManifestError
from .versioning import SemanticVersion

_ID_RE = re.compile(r"^[a-z][a-z0-9_.-]{2,127}$")
_TOKEN_RE = re.compile(r"^[a-z][a-z0-9_.-]{0,127}$")
_ENTRYPOINT_RE = re.compile(
    r"^[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*:"
    r"[A-Za-z_][A-Za-z0-9_]*(?:\.[A-Za-z_][A-Za-z0-9_]*)*$"
)
_DEP_RE = re.compile(r"^([a-z][a-z0-9_.-]{2,127})(.*)$")


class ExtensionKind(str, Enum):
    BACKEND = "backend"
    BRIDGE = "bridge"
    DATA_CONNECTOR = "data_connector"
    FEATURE_EXTRACTOR = "feature_extractor"
    EVALUATOR = "evaluator"
    REPORTER = "reporter"
    APPLICATION = "application"
    MIXED = "mixed"


class ExtensionLifecycle(str, Enum):
    EXPERIMENTAL = "experimental"
    ACTIVE = "active"
    DEPRECATED = "deprecated"


def _kind(value: ExtensionKind | str) -> ExtensionKind:
    if isinstance(value, ExtensionKind):
        return value
    try:
        return ExtensionKind(str(value).strip().lower())
    except ValueError as exc:
        raise ExtensionManifestError(f"unsupported extension kind {value!r}") from exc


def _lifecycle(value: ExtensionLifecycle | str) -> ExtensionLifecycle:
    if isinstance(value, ExtensionLifecycle):
        return value
    try:
        return ExtensionLifecycle(str(value).strip().lower())
    except ValueError as exc:
        raise ExtensionManifestError(f"unsupported extension lifecycle {value!r}") from exc


def _token(value: object, *, field_name: str) -> str:
    item = str(value).strip().lower()
    if not _TOKEN_RE.fullmatch(item):
        raise ExtensionManifestError(f"invalid {field_name} token {value!r}")
    return item


def _freeze_json(
    value: Any,
    *,
    field_name: str,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> Any:
    if _depth > 64:
        raise ExtensionManifestError(f"{field_name} nesting exceeds the supported depth")
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtensionManifestError(f"{field_name} may not contain NaN or Infinity")
        return value
    if isinstance(value, (Mapping, list, tuple)):
        seen = set() if _seen is None else _seen
        marker = id(value)
        if marker in seen:
            raise ExtensionManifestError(f"{field_name} may not contain cyclic data")
        seen.add(marker)
        try:
            if isinstance(value, Mapping):
                frozen: dict[str, Any] = {}
                for key, item in value.items():
                    if not isinstance(key, str):
                        raise ExtensionManifestError(f"{field_name} mapping keys must be strings")
                    frozen[key] = _freeze_json(
                        item, field_name=field_name, _seen=seen, _depth=_depth + 1
                    )
                return MappingProxyType(frozen)
            return tuple(
                _freeze_json(item, field_name=field_name, _seen=seen, _depth=_depth + 1)
                for item in value
            )
        finally:
            seen.remove(marker)
    raise ExtensionManifestError(
        f"{field_name} must be finite JSON-compatible data, not {type(value).__name__}"
    )


def _thaw_json(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _thaw_json(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return [_thaw_json(item) for item in value]
    return value


@dataclass(frozen=True, slots=True)
class ExtensionDependency:
    extension_id: str
    version: str = "*"

    def __post_init__(self) -> None:
        extension_id = str(self.extension_id).strip().lower()
        if not _ID_RE.fullmatch(extension_id):
            raise ExtensionManifestError(f"invalid dependency extension_id {self.extension_id!r}")
        spec = str(self.version).strip() or "*"
        # Validate every referenced version eagerly.
        if spec != "*":
            from .versioning import version_satisfies

            # A version that satisfies itself is not required; this call is only syntax validation.
            # Use a harmless concrete version and tolerate False, but propagate malformed syntax.
            version_satisfies("0.0.0", spec)
        object.__setattr__(self, "extension_id", extension_id)
        object.__setattr__(self, "version", spec)

    @classmethod
    def parse(cls, value: "ExtensionDependency | str") -> "ExtensionDependency":
        if isinstance(value, cls):
            return value
        raw = str(value).strip()
        match = _DEP_RE.fullmatch(raw)
        if match is None:
            raise ExtensionManifestError(f"invalid dependency declaration {value!r}")
        extension_id = match.group(1)
        spec = match.group(2).strip() or "*"
        return cls(extension_id=extension_id, version=spec)

    def to_dict(self) -> dict[str, str]:
        return {"extension_id": self.extension_id, "version": self.version}


@dataclass(frozen=True, slots=True)
class ExtensionManifest:
    """Immutable, content-addressed description of a trusted SolverPilot extension."""

    extension_id: str
    name: str
    version: str
    kind: ExtensionKind | str
    description: str
    capabilities: tuple[str, ...] = ()
    dependencies: tuple[ExtensionDependency | str, ...] = ()
    entrypoint: str | None = None
    lifecycle: ExtensionLifecycle | str = ExtensionLifecycle.EXPERIMENTAL
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        extension_id = str(self.extension_id).strip().lower()
        if not _ID_RE.fullmatch(extension_id):
            raise ExtensionManifestError(
                "extension_id must start with a lowercase letter, be 3-128 characters, "
                "and contain only lowercase letters, digits, '.', '_' or '-'"
            )
        name = str(self.name).strip()
        description = str(self.description).strip()
        if not name:
            raise ExtensionManifestError("name must be non-empty")
        if not description:
            raise ExtensionManifestError("description must be non-empty")
        version = str(self.version).strip()
        SemanticVersion.parse(version)
        capabilities: list[str] = []
        for value in self.capabilities:
            item = _token(value, field_name="capability")
            if item not in capabilities:
                capabilities.append(item)
        dependencies: list[ExtensionDependency] = []
        seen: set[str] = set()
        for value in self.dependencies:
            dep = ExtensionDependency.parse(value)
            if dep.extension_id == extension_id:
                raise ExtensionManifestError("an extension may not depend on itself")
            if dep.extension_id in seen:
                raise ExtensionManifestError(f"duplicate dependency {dep.extension_id!r}")
            seen.add(dep.extension_id)
            dependencies.append(dep)
        entrypoint = None if self.entrypoint is None else str(self.entrypoint).strip()
        if entrypoint == "":
            entrypoint = None
        if entrypoint is not None and not _ENTRYPOINT_RE.fullmatch(entrypoint):
            raise ExtensionManifestError("entrypoint must use the form 'package.module:Attribute'")
        object.__setattr__(self, "extension_id", extension_id)
        object.__setattr__(self, "name", name)
        object.__setattr__(self, "version", version)
        object.__setattr__(self, "kind", _kind(self.kind))
        object.__setattr__(self, "description", description)
        object.__setattr__(self, "capabilities", tuple(capabilities))
        object.__setattr__(self, "dependencies", tuple(dependencies))
        object.__setattr__(self, "entrypoint", entrypoint)
        object.__setattr__(self, "lifecycle", _lifecycle(self.lifecycle))
        object.__setattr__(self, "metadata", _freeze_json(self.metadata, field_name="metadata"))

    def to_dict(self) -> dict[str, Any]:
        return {
            "extension_id": self.extension_id,
            "name": self.name,
            "version": self.version,
            "kind": self.kind.value,
            "description": self.description,
            "capabilities": list(self.capabilities),
            "dependencies": [dep.to_dict() for dep in self.dependencies],
            "entrypoint": self.entrypoint,
            "lifecycle": self.lifecycle.value,
            "metadata": _thaw_json(self.metadata),
        }

    @property
    def sha256(self) -> str:
        raw = json.dumps(
            self.to_dict(), sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
        ).encode("utf-8")
        return hashlib.sha256(raw).hexdigest()
