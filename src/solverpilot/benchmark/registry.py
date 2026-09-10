from __future__ import annotations

from solverpilot.backends import BackendRegistry, BundledOSQPCAPIBackend, BundledHighsCAPIBackend
from solverpilot.runtime.auto import builtin_backend_candidates


def benchmark_registry(backends: tuple[str, ...]) -> BackendRegistry:
    backends = tuple(b for b in backends if b != "@auto")
    if not backends:
        raise ValueError("benchmark registry needs at least one explicit backend; @auto is evaluated over that portfolio")
    candidates = (*builtin_backend_candidates(), BundledOSQPCAPIBackend(), BundledHighsCAPIBackend())
    by_name = {b.manifest.name: b for b in candidates}
    unknown = sorted(set(backends) - set(by_name))
    if unknown:
        raise KeyError(f"unknown benchmark backends: {unknown}")
    registry = BackendRegistry()
    unavailable = []
    unverified = []
    for name in backends:
        backend = by_name[name]
        if not backend.is_available():
            unavailable.append(name)
            continue
        verified_bridge_versions = backend.manifest.metadata.get("verified_bridge_package_versions")
        bridge_version = backend.manifest.metadata.get("bridge_package_version")
        if verified_bridge_versions is not None and bridge_version not in set(verified_bridge_versions):
            unverified.append((name, f"bridge={bridge_version}", tuple(verified_bridge_versions)))
            continue
        verified_solver_versions = backend.manifest.metadata.get("verified_solver_versions")
        if verified_solver_versions is not None and backend.manifest.version not in set(verified_solver_versions):
            unverified.append((name, f"solver={backend.manifest.version}", tuple(verified_solver_versions)))
            continue
        registry.register(backend)
    if unavailable:
        raise RuntimeError(f"requested backends are unavailable: {unavailable}")
    if unverified:
        raise RuntimeError(f"requested verification bridges use unverified package versions: {unverified}")
    return registry
