from __future__ import annotations

from .base import Backend
from solverpilot.exceptions import UnknownBackendError


class BackendRegistry:
    def __init__(self) -> None:
        self._backends: dict[str, Backend] = {}

    def register(self, backend: Backend) -> None:
        name = backend.manifest.name
        if name in self._backends:
            raise ValueError(f"backend already registered: {name}")
        self._backends[name] = backend

    def get(self, name: str) -> Backend:
        try:
            return self._backends[name]
        except KeyError as exc:
            raise UnknownBackendError(f"unknown backend: {name}") from exc

    def all(self) -> tuple[Backend, ...]:
        return tuple(self._backends.values())

    def available(self) -> tuple[Backend, ...]:
        return tuple(b for b in self._backends.values() if b.is_available())

    def names(self) -> tuple[str, ...]:
        return tuple(sorted(self._backends))

    def capability_manifest_v2(self, name: str, *, verify: bool = False):
        from solverpilot.capabilities import resolve_backend_capabilities
        return resolve_backend_capabilities(self.get(name), verify=verify)

    def capability_manifests_v2(self, *, verify: bool = False):
        from solverpilot.capabilities import resolve_backend_capabilities
        return tuple(resolve_backend_capabilities(b, verify=verify) for b in self.all())
