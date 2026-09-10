from __future__ import annotations

from dataclasses import dataclass

from solverpilot.capabilities.v2 import BackendCapabilityManifestV2, CapabilityKey

from .registry import default_bridge_registry
from .types import BridgePolicy, ExactnessClass


@dataclass(frozen=True, slots=True)
class BridgePathDecision:
    semantic_kind: str
    selected_path: str
    native_usable: bool
    considered_paths: tuple[str, ...] = ()
    rejected_paths: tuple[str, ...] = ()


def choose_indicator_path(
    capabilities: BackendCapabilityManifestV2 | None,
    *,
    native_transport_available: bool,
    policy: BridgePolicy,
) -> BridgePathDecision:
    specs = default_bridge_registry().candidates("indicator", policy=policy)
    considered = tuple(spec.bridge_id for spec in specs)
    native_usable = False
    rejected: list[str] = []
    if capabilities is not None:
        check = capabilities.check(CapabilityKey.CONSTRAINT_INDICATOR)
        native_usable = check.usable
        if check.usable and not native_transport_available:
            rejected.append("native indicator capability verified but current execution IR has no native-indicator transport")
        elif not check.usable:
            rejected.append(f"native indicator path unavailable: {check.reason}")
    else:
        rejected.append("native indicator path unavailable: no backend capability manifest supplied")

    if native_usable and native_transport_available:
        return BridgePathDecision("indicator", "native", True, considered, tuple(rejected))
    if not policy.permits(ExactnessClass.EXACT_EQUIVALENT):
        raise RuntimeError("bridge policy rejects exact indicator bridge")
    return BridgePathDecision("indicator", "certified_big_m", native_usable, considered, tuple(rejected))
