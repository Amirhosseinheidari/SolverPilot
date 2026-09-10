from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from threading import RLock
from enum import Enum
import math
from collections.abc import Mapping
from typing import Iterable

from solverpilot.backends import BackendRegistry
from solverpilot.backends.base import Backend
from solverpilot.bridges import BridgeRegistry, BridgeSpec
from solverpilot.capabilities import BackendManifest

from .errors import (
    ExtensionActivationError,
    ExtensionDependencyError,
    ExtensionRegistrationError,
    ExtensionRegistryFrozenError,
    ExtensionValidationError,
)
from .manifest import ExtensionKind, ExtensionManifest
from .model import (
    ExportKind,
    Extension,
    ExtensionActivationPlan,
    ExtensionContributions,
    ExtensionSelfCheck,
    NamedExtensionExport,
    StagedExtensionContribution,
)
from .versioning import version_satisfies


@dataclass(frozen=True, slots=True)
class _Registered:
    extension: Extension
    manifest: ExtensionManifest


@dataclass(frozen=True, slots=True)
class _IssuedPlan:
    plan: ExtensionActivationPlan
    fingerprint: str
    generation: int
    backend_target: BackendRegistry
    bridge_target: BridgeRegistry
    backend_target_fingerprint: str
    bridge_target_fingerprint: str


def _manifest_snapshot(extension: Extension) -> ExtensionManifest:
    manifest = extension.manifest
    if not isinstance(manifest, ExtensionManifest):
        raise ExtensionValidationError("extension.manifest must be an ExtensionManifest")
    # Reconstruct from public data so nested immutable state is detached from caller-owned objects.
    data = manifest.to_dict()
    data["dependencies"] = tuple(
        f"{item['extension_id']}{'' if item['version'] == '*' else item['version']}"
        for item in data["dependencies"]
    )
    return ExtensionManifest(**data)


def _run_self_check(extension: Extension) -> ExtensionSelfCheck:
    try:
        check = extension.self_check()
    except Exception as exc:
        raise ExtensionValidationError(f"extension self-check raised: {exc}") from exc
    if not isinstance(check, ExtensionSelfCheck):
        raise ExtensionValidationError("extension.self_check() must return ExtensionSelfCheck")
    if not check.ok:
        raise ExtensionValidationError("extension self-check did not pass")
    return check


def _bridge_key(spec: BridgeSpec) -> str:
    return f"{spec.bridge_id}@{spec.bridge_version}"


def _observable(
    value: object,
    *,
    _seen: set[int] | None = None,
    _depth: int = 0,
) -> object:
    """Create a deterministic in-process snapshot for drift detection.

    The snapshot is not a public serialization format. Known containers/enums/scalars are
    normalized structurally; unknown objects are bound by type, repr, and identity. Cyclic,
    excessively deep, or non-finite container state fails closed instead of producing an
    ambiguous activation fingerprint.
    """

    if _depth > 64:
        raise ExtensionValidationError("backend observable state exceeds the supported depth")
    if isinstance(value, Enum):
        return {"enum": f"{type(value).__module__}.{type(value).__qualname__}", "value": value.value}
    if value is None or isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ExtensionValidationError("backend observable state may not contain NaN or Infinity")
        return value
    if isinstance(value, (Mapping, tuple, list, set, frozenset)):
        seen = set() if _seen is None else _seen
        marker = id(value)
        if marker in seen:
            raise ExtensionValidationError("backend observable state may not contain cyclic data")
        seen.add(marker)
        try:
            if isinstance(value, Mapping):
                pairs = [
                    [
                        _observable(key, _seen=seen, _depth=_depth + 1),
                        _observable(item, _seen=seen, _depth=_depth + 1),
                    ]
                    for key, item in value.items()
                ]
                pairs.sort(
                    key=lambda pair: json.dumps(
                        pair[0], sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                    )
                )
                return {"mapping": pairs}
            items = [_observable(item, _seen=seen, _depth=_depth + 1) for item in value]
            if isinstance(value, (set, frozenset)):
                items.sort(
                    key=lambda item: json.dumps(
                        item, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
                    )
                )
            return {type(value).__name__: items}
        finally:
            seen.remove(marker)
    return {
        "type": f"{type(value).__module__}.{type(value).__qualname__}",
        "repr": repr(value),
        "id": id(value),
    }


def _backend_state(backend: Backend) -> object:
    try:
        manifest = backend.manifest
    except Exception as exc:
        raise ExtensionValidationError(f"backend contribution manifest raised: {exc}") from exc
    if not isinstance(manifest, BackendManifest):
        raise ExtensionValidationError("backend contribution manifest must be BackendManifest")
    return {
        "object_id": id(backend),
        "name": manifest.name,
        "version": manifest.version,
        "capabilities": _observable(manifest.capabilities),
        "metadata": _observable(manifest.metadata),
    }


def _backend_target_fingerprint(registry: BackendRegistry) -> str:
    payload = [_backend_state(backend) for backend in registry.all()]
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False).encode()
    return hashlib.sha256(raw).hexdigest()


def _bridge_target_fingerprint(registry: BridgeRegistry) -> str:
    payload = [
        (
            spec.bridge_id,
            spec.bridge_version,
            spec.source_kind,
            spec.exactness.value,
            spec.priority,
            spec.description,
        )
        for spec in registry.specs
    ]
    raw = json.dumps(payload, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(raw).hexdigest()


def _plan_fingerprint(plan: ExtensionActivationPlan) -> str:
    payload: dict[str, object] = {
        "generation": plan.generation,
        "order": list(plan.extension_order),
        "targets": {
            "backend_names": list(plan.target_backend_names),
            "bridge_keys": list(plan.target_bridge_keys),
        },
        "staged": [],
    }
    staged_payload: list[dict[str, object]] = []
    for item in plan.staged:
        staged_payload.append(
            {
                "extension_id": item.extension_id,
                "extension_version": item.extension_version,
                "backends": [_backend_state(backend) for backend in item.backends],
                "bridges": [
                    (
                        spec.bridge_id,
                        spec.bridge_version,
                        spec.source_kind,
                        spec.exactness.value,
                        spec.priority,
                        spec.description,
                        id(spec),
                    )
                    for spec in item.bridges
                ],
                "exports": [(export.key, id(export.value), id(export)) for export in item.exports],
            }
        )
    payload["staged"] = staged_payload
    raw = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False
    ).encode()
    return hashlib.sha256(raw).hexdigest()


def _validate_target_registries(
    backend_registry: BackendRegistry, bridge_registry: BridgeRegistry
) -> None:
    if not isinstance(backend_registry, BackendRegistry):
        raise ExtensionActivationError("backend_registry must be SolverPilot BackendRegistry")
    if not isinstance(bridge_registry, BridgeRegistry):
        raise ExtensionActivationError("bridge_registry must be SolverPilot BridgeRegistry")


class ExtensionManager:
    """Explicit, freezeable manager for trusted SolverPilot extension packages.

    The manager does not replace SolverPilot's canonical BackendRegistry or BridgeRegistry.
    Backend/bridge contributions are staged first and are applied only to caller-supplied canonical
    registries after all manager-side validation succeeds. Third-party extensions execute in-process
    and are trusted code; this module does not provide a sandbox.
    """

    def __init__(self, extensions: Iterable[Extension] | None = None) -> None:
        self._lock = RLock()
        self._activation_lock = RLock()
        self._registered: dict[str, _Registered] = {}
        self._exports: dict[str, NamedExtensionExport] = {}
        self._frozen = False
        self._generation = 0
        self._issued_plans: dict[int, _IssuedPlan] = {}
        if extensions is not None:
            self.register_many(extensions)

    @property
    def frozen(self) -> bool:
        with self._lock:
            return self._frozen

    @property
    def generation(self) -> int:
        with self._lock:
            return self._generation

    def register(self, extension: Extension) -> None:
        self.register_many((extension,))

    def register_many(self, extensions: Iterable[Extension]) -> None:
        incoming = tuple(extensions)
        if not incoming:
            return
        prepared: list[_Registered] = []
        seen: set[str] = set()
        # Extension-controlled properties/hooks execute outside the manager lock.
        for extension in incoming:
            if not isinstance(extension, Extension):
                raise ExtensionValidationError(
                    "extension must provide manifest, self_check(), and contributions()"
                )
            manifest = _manifest_snapshot(extension)
            _run_self_check(extension)
            # Ensure the object did not change identity during validation.
            if _manifest_snapshot(extension) != manifest:
                raise ExtensionValidationError(
                    f"extension {manifest.extension_id!r} changed its manifest during registration"
                )
            if manifest.extension_id in seen:
                raise ExtensionRegistrationError(
                    f"duplicate extension id in registration batch: {manifest.extension_id}"
                )
            seen.add(manifest.extension_id)
            prepared.append(_Registered(extension=extension, manifest=manifest))
        with self._lock:
            if self._frozen:
                raise ExtensionRegistryFrozenError("extension manager is frozen")
            collisions = sorted(seen.intersection(self._registered))
            if collisions:
                raise ExtensionRegistrationError(
                    f"extension ids already registered: {', '.join(collisions)}"
                )
            for item in prepared:
                self._registered[item.manifest.extension_id] = item
            self._generation += 1

    def get(self, extension_id: str) -> Extension:
        key = str(extension_id).strip().lower()
        with self._lock:
            try:
                return self._registered[key].extension
            except KeyError as exc:
                raise ExtensionRegistrationError(f"unknown extension id {extension_id!r}") from exc

    def list_extension_ids(self) -> tuple[str, ...]:
        with self._lock:
            return tuple(sorted(self._registered))

    def manifests(self) -> tuple[ExtensionManifest, ...]:
        with self._lock:
            return tuple(self._registered[key].manifest for key in sorted(self._registered))

    def _dependency_analysis(
        self, entries: tuple[_Registered, ...]
    ) -> tuple[dict[str, tuple[str, ...]], tuple[str, ...]]:
        manifests = {item.manifest.extension_id: item.manifest for item in entries}
        missing: dict[str, tuple[str, ...]] = {}
        graph: dict[str, set[str]] = {}
        reverse: dict[str, set[str]] = {key: set() for key in manifests}
        indegree: dict[str, int] = {key: 0 for key in manifests}
        for extension_id, manifest in manifests.items():
            deps: set[str] = set()
            unresolved: list[str] = []
            for dep in manifest.dependencies:
                target = manifests.get(dep.extension_id)
                if target is None:
                    unresolved.append(f"{dep.extension_id}{'' if dep.version == '*' else dep.version}")
                    continue
                if not version_satisfies(target.version, dep.version):
                    raise ExtensionDependencyError(
                        f"extension {extension_id!r} requires {dep.extension_id}{dep.version}, "
                        f"but registered version is {target.version}"
                    )
                deps.add(dep.extension_id)
            if unresolved:
                missing[extension_id] = tuple(sorted(unresolved))
            graph[extension_id] = deps
            indegree[extension_id] = len(deps)
            for dep_id in deps:
                reverse[dep_id].add(extension_id)
        if missing:
            details = "; ".join(f"{key}: {', '.join(values)}" for key, values in sorted(missing.items()))
            raise ExtensionDependencyError(f"unresolved extension dependencies: {details}")
        ready = sorted(key for key, degree in indegree.items() if degree == 0)
        order: list[str] = []
        while ready:
            current = ready.pop(0)
            order.append(current)
            for consumer in sorted(reverse[current]):
                indegree[consumer] -= 1
                if indegree[consumer] == 0:
                    ready.append(consumer)
                    ready.sort()
        if len(order) != len(manifests):
            cyclic = sorted(key for key, degree in indegree.items() if degree > 0)
            raise ExtensionDependencyError(
                f"extension dependency graph contains a cycle involving: {', '.join(cyclic)}"
            )
        return {key: tuple(sorted(values)) for key, values in graph.items()}, tuple(order)

    def freeze(self) -> None:
        while True:
            with self._lock:
                if self._frozen:
                    return
                generation = self._generation
                entries = tuple(self._registered.values())
            # Do not invoke extension-owned properties while holding the manager lock.
            for item in entries:
                if _manifest_snapshot(item.extension) != item.manifest:
                    raise ExtensionValidationError(
                        f"extension {item.manifest.extension_id!r} manifest drifted after registration"
                    )
            self._dependency_analysis(entries)
            with self._lock:
                if self._generation != generation:
                    continue
                self._frozen = True
                self._generation += 1
                self._issued_plans.clear()
                return

    def dependency_order(self) -> tuple[str, ...]:
        with self._lock:
            entries = tuple(self._registered.values())
        return self._dependency_analysis(entries)[1]

    def _validate_contributions(
        self, manifest: ExtensionManifest, contributions: ExtensionContributions
    ) -> StagedExtensionContribution:
        if not isinstance(contributions, ExtensionContributions):
            raise ExtensionValidationError(
                f"extension {manifest.extension_id!r} contributions() must return ExtensionContributions"
            )
        backend_names: set[str] = set()
        for backend in contributions.backends:
            if not isinstance(backend, Backend):
                raise ExtensionValidationError("backend contribution does not satisfy Backend protocol")
            backend_manifest = backend.manifest
            if not isinstance(backend_manifest, BackendManifest):
                raise ExtensionValidationError("backend contribution manifest must be BackendManifest")
            name = str(backend_manifest.name).strip()
            if not name:
                raise ExtensionValidationError("backend contribution name must be non-empty")
            if name in backend_names:
                raise ExtensionValidationError(f"duplicate backend contribution {name!r}")
            backend_names.add(name)
        bridge_keys: set[str] = set()
        for bridge in contributions.bridges:
            if not isinstance(bridge, BridgeSpec):
                raise ExtensionValidationError("bridge contribution must be BridgeSpec")
            from solverpilot.bridges import ExactnessClass

            if not bridge.bridge_id or not bridge.bridge_version or not bridge.source_kind or not bridge.description:
                raise ExtensionValidationError("bridge contribution identifiers and description must be non-empty")
            if not isinstance(bridge.exactness, ExactnessClass):
                raise ExtensionValidationError("bridge contribution exactness must be ExactnessClass")
            if isinstance(bridge.priority, bool) or not isinstance(bridge.priority, int):
                raise ExtensionValidationError("bridge contribution priority must be an integer")
            key = _bridge_key(bridge)
            if key in bridge_keys:
                raise ExtensionValidationError(f"duplicate bridge contribution {key!r}")
            bridge_keys.add(key)
        export_keys: set[str] = set()
        for export in contributions.exports:
            if not isinstance(export, NamedExtensionExport):
                raise ExtensionValidationError("export contribution must be NamedExtensionExport")
            if export.key in export_keys:
                raise ExtensionValidationError(f"duplicate named export {export.key!r}")
            export_keys.add(export.key)

        if manifest.kind is not ExtensionKind.MIXED:
            actual_kinds: set[ExtensionKind] = set()
            if contributions.backends:
                actual_kinds.add(ExtensionKind.BACKEND)
            if contributions.bridges:
                actual_kinds.add(ExtensionKind.BRIDGE)
            export_kind_map = {
                ExportKind.DATA_CONNECTOR: ExtensionKind.DATA_CONNECTOR,
                ExportKind.FEATURE_EXTRACTOR: ExtensionKind.FEATURE_EXTRACTOR,
                ExportKind.EVALUATOR: ExtensionKind.EVALUATOR,
                ExportKind.REPORTER: ExtensionKind.REPORTER,
                ExportKind.APPLICATION: ExtensionKind.APPLICATION,
            }
            actual_kinds.update(export_kind_map[item.kind] for item in contributions.exports)
            unexpected = sorted(kind.value for kind in actual_kinds if kind is not manifest.kind)
            if unexpected:
                raise ExtensionValidationError(
                    f"extension {manifest.extension_id!r} declares kind {manifest.kind.value!r} "
                    f"but contributes: {', '.join(unexpected)}; use kind='mixed' for mixed contributions"
                )

        return StagedExtensionContribution(
            extension_id=manifest.extension_id,
            extension_version=manifest.version,
            backends=tuple(contributions.backends),
            bridges=tuple(contributions.bridges),
            exports=tuple(contributions.exports),
        )

    def plan_activation(
        self,
        *,
        backend_registry: BackendRegistry,
        bridge_registry: BridgeRegistry,
    ) -> ExtensionActivationPlan:
        _validate_target_registries(backend_registry, bridge_registry)
        with self._lock:
            if not self._frozen:
                raise ExtensionActivationError("freeze the extension manager before activation planning")
            generation = self._generation
            entries_by_id = dict(self._registered)
            existing_export_keys = set(self._exports)
        order = self._dependency_analysis(tuple(entries_by_id.values()))[1]
        staged: list[StagedExtensionContribution] = []
        all_backend_names: set[str] = set()
        all_bridge_keys: set[str] = set()
        all_export_keys: set[str] = set()
        for extension_id in order:
            item = entries_by_id[extension_id]
            if _manifest_snapshot(item.extension) != item.manifest:
                raise ExtensionValidationError(f"extension {extension_id!r} manifest drifted before staging")
            _run_self_check(item.extension)
            try:
                contributions = item.extension.contributions()
            except Exception as exc:
                raise ExtensionValidationError(
                    f"extension {extension_id!r} contributions() raised: {exc}"
                ) from exc
            staged_item = self._validate_contributions(item.manifest, contributions)
            if _manifest_snapshot(item.extension) != item.manifest:
                raise ExtensionValidationError(f"extension {extension_id!r} manifest drifted during staging")
            for backend in staged_item.backends:
                name = backend.manifest.name
                if name in all_backend_names:
                    raise ExtensionActivationError(f"duplicate staged backend name {name!r}")
                all_backend_names.add(name)
            for bridge in staged_item.bridges:
                key = _bridge_key(bridge)
                if key in all_bridge_keys:
                    raise ExtensionActivationError(f"duplicate staged bridge key {key!r}")
                all_bridge_keys.add(key)
            for export in staged_item.exports:
                if export.key in all_export_keys:
                    raise ExtensionActivationError(f"duplicate staged export key {export.key!r}")
                all_export_keys.add(export.key)
            staged.append(staged_item)
        target_backend_names = tuple(sorted(backend_registry.names()))
        target_bridge_keys = tuple(sorted(_bridge_key(spec) for spec in bridge_registry.specs))
        collisions = sorted(all_backend_names.intersection(target_backend_names))
        if collisions:
            raise ExtensionActivationError(f"backend target collisions: {', '.join(collisions)}")
        bridge_collisions = sorted(all_bridge_keys.intersection(target_bridge_keys))
        if bridge_collisions:
            raise ExtensionActivationError(f"bridge target collisions: {', '.join(bridge_collisions)}")
        export_collisions = sorted(all_export_keys.intersection(existing_export_keys))
        if export_collisions:
            raise ExtensionActivationError(f"export target collisions: {', '.join(export_collisions)}")
        provisional = ExtensionActivationPlan(
            generation=generation,
            extension_order=order,
            staged=tuple(staged),
            target_backend_names=target_backend_names,
            target_bridge_keys=target_bridge_keys,
            plan_sha256="",
        )
        fingerprint = _plan_fingerprint(provisional)
        plan = ExtensionActivationPlan(
            generation=provisional.generation,
            extension_order=provisional.extension_order,
            staged=provisional.staged,
            target_backend_names=provisional.target_backend_names,
            target_bridge_keys=provisional.target_bridge_keys,
            plan_sha256=fingerprint,
        )
        issued = _IssuedPlan(
            plan=plan,
            fingerprint=_plan_fingerprint(plan),
            generation=generation,
            backend_target=backend_registry,
            bridge_target=bridge_registry,
            backend_target_fingerprint=_backend_target_fingerprint(backend_registry),
            bridge_target_fingerprint=_bridge_target_fingerprint(bridge_registry),
        )
        with self._lock:
            if not self._frozen or self._generation != generation:
                raise ExtensionActivationError("extension manager changed while activation was staged")
            self._issued_plans[id(plan)] = issued
        return plan

    def apply_activation(
        self,
        plan: ExtensionActivationPlan,
        *,
        backend_registry: BackendRegistry,
        bridge_registry: BridgeRegistry,
    ) -> None:
        if not isinstance(plan, ExtensionActivationPlan):
            raise ExtensionActivationError("plan must be ExtensionActivationPlan")
        _validate_target_registries(backend_registry, bridge_registry)

        # Serialize applies without holding the central query/registration lock while touching
        # extension-provided backend objects. Direct external mutation of caller-owned canonical
        # registries during this method remains outside the manager's synchronization contract.
        with self._activation_lock:
            current_plan_fingerprint = _plan_fingerprint(plan)
            backend_target_fingerprint = _backend_target_fingerprint(backend_registry)
            bridge_target_fingerprint = _bridge_target_fingerprint(bridge_registry)
            with self._lock:
                issued = self._issued_plans.get(id(plan))
                if issued is None or issued.plan is not plan:
                    raise ExtensionActivationError(
                        "activation plan was not issued by this manager or was consumed"
                    )
                if issued.backend_target is not backend_registry or issued.bridge_target is not bridge_registry:
                    raise ExtensionActivationError(
                        "activation plan is bound to different canonical registry objects"
                    )
                if plan.generation != self._generation or issued.generation != self._generation:
                    raise ExtensionActivationError("activation plan generation is stale")
                if plan.plan_sha256 != current_plan_fingerprint or issued.fingerprint != current_plan_fingerprint:
                    raise ExtensionActivationError("activation plan changed after staging")
                if backend_target_fingerprint != issued.backend_target_fingerprint:
                    raise ExtensionActivationError("backend registry changed after activation planning")
                if bridge_target_fingerprint != issued.bridge_target_fingerprint:
                    raise ExtensionActivationError("bridge registry changed after activation planning")
                staged_exports = tuple(export for item in plan.staged for export in item.exports)
                for export in staged_exports:
                    if export.key in self._exports:
                        raise ExtensionActivationError(f"export target collision: {export.key}")
                # Reserve single-use ownership before any caller-owned registry is mutated.
                del self._issued_plans[id(plan)]

            staged_backends = tuple(backend for item in plan.staged for backend in item.backends)
            staged_bridges = tuple(bridge for item in plan.staged for bridge in item.bridges)

            # Preview with canonical registry implementations. This verifies every built-in
            # registration rule before mutating caller-owned targets. Under the documented
            # no-concurrent-direct-mutation precondition, the subsequent canonical register calls
            # cannot introduce a new duplicate/collision.
            preview_backends = BackendRegistry()
            for backend in backend_registry.all():
                preview_backends.register(backend)
            for backend in staged_backends:
                preview_backends.register(backend)
            preview_bridges = BridgeRegistry()
            for spec in bridge_registry.specs:
                preview_bridges.register(spec)
            for spec in staged_bridges:
                preview_bridges.register(spec)

            try:
                for backend in staged_backends:
                    backend_registry.register(backend)
                for spec in staged_bridges:
                    bridge_registry.register(spec)
            except Exception as exc:
                # This path should only be reachable if caller-owned registries are concurrently
                # mutated or a non-canonical registry violates the previewed registration contract.
                raise ExtensionActivationError(
                    "canonical registry activation failed after prevalidation; the activation plan "
                    "is consumed and target registries must be inspected before retrying"
                ) from exc

            with self._lock:
                for export in staged_exports:
                    if export.key in self._exports:
                        # Same-manager applies are serialized, so this is a defensive invariant.
                        raise ExtensionActivationError(f"export target collision: {export.key}")
                    self._exports[export.key] = export

    def list_exports(self, *, kind: ExportKind | str | None = None) -> tuple[NamedExtensionExport, ...]:
        normalized = None if kind is None else (kind if isinstance(kind, ExportKind) else ExportKind(str(kind).strip().lower()))
        with self._lock:
            values = tuple(self._exports[key] for key in sorted(self._exports))
        if normalized is None:
            return values
        return tuple(item for item in values if item.kind is normalized)

    def get_export(self, kind: ExportKind | str, name: str) -> object:
        normalized_kind = kind if isinstance(kind, ExportKind) else ExportKind(str(kind).strip().lower())
        key = f"{normalized_kind.value}:{str(name).strip().lower()}"
        with self._lock:
            try:
                return self._exports[key].value
            except KeyError as exc:
                raise ExtensionRegistrationError(f"unknown extension export {key!r}") from exc
