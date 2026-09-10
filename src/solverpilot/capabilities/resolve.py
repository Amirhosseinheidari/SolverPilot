from __future__ import annotations

from typing import Protocol, runtime_checkable

from .conformance import conform_backend_problem_classes
from .v2 import BackendCapabilityManifestV2, project_legacy_manifest


@runtime_checkable
class BackendCapabilityProviderV2(Protocol):
    @property
    def capability_manifest_v2(self) -> BackendCapabilityManifestV2: ...


def resolve_backend_capabilities(backend, *, verify: bool = False) -> BackendCapabilityManifestV2:
    """Resolve a backend's v2 manifest without upgrading unverified declarations.

    Future adapters may expose an explicit ``capability_manifest_v2``. Current Track-M
    adapters are projected conservatively from v1. ``verify=True`` runs the P3 native
    problem-class conformance kit and upgrades only claims directly exercised by it.
    """
    explicit = getattr(backend, "capability_manifest_v2", None)
    if explicit is not None:
        manifest = explicit() if callable(explicit) else explicit
        if not isinstance(manifest, BackendCapabilityManifestV2):
            raise TypeError("capability_manifest_v2 must be BackendCapabilityManifestV2")
        return manifest
    if verify:
        return conform_backend_problem_classes(backend).manifest
    return project_legacy_manifest(backend.manifest)
