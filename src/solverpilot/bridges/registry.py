from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from .types import BridgePolicy, ExactnessClass


@dataclass(frozen=True, slots=True)
class BridgeSpec:
    bridge_id: str
    bridge_version: str
    source_kind: str
    exactness: ExactnessClass
    priority: int
    description: str


@dataclass(frozen=True, slots=True)
class BridgeCandidate:
    spec: BridgeSpec
    applicable: bool
    reason: str


class BridgeRegistry:
    """Small deterministic registry for P4 reformulation candidates.

    The registry is intentionally evidence-free: it orders safe semantic paths by a
    deterministic structural priority. Runtime performance evidence remains Track M's
    responsibility and is not smuggled into P4 compilation decisions.
    """

    def __init__(self) -> None:
        self._specs: dict[str, BridgeSpec] = {}

    def register(self, spec: BridgeSpec) -> None:
        key = f"{spec.bridge_id}@{spec.bridge_version}"
        if key in self._specs:
            raise ValueError(f"duplicate bridge registration {key}")
        self._specs[key] = spec

    def candidates(self, source_kind: str, *, policy: BridgePolicy) -> tuple[BridgeSpec, ...]:
        items = [s for s in self._specs.values() if s.source_kind == source_kind and policy.permits(s.exactness)]
        return tuple(sorted(items, key=lambda s: (s.priority, s.bridge_id, s.bridge_version)))

    @property
    def specs(self) -> tuple[BridgeSpec, ...]:
        return tuple(sorted(self._specs.values(), key=lambda s: (s.source_kind, s.priority, s.bridge_id, s.bridge_version)))


def default_bridge_registry() -> BridgeRegistry:
    reg = BridgeRegistry()
    reg.register(BridgeSpec(
        bridge_id="solverpilot.bridge.indicator.native",
        bridge_version="1.0",
        source_kind="indicator",
        exactness=ExactnessClass.EXACT_EQUIVALENT,
        priority=0,
        description="Preserve a native indicator constraint when both backend capability and execution transport are verified.",
    ))
    reg.register(BridgeSpec(
        bridge_id="solverpilot.bridge.indicator.fixed",
        bridge_version="1.0",
        source_kind="indicator",
        exactness=ExactnessClass.EXACT_EQUIVALENT,
        priority=10,
        description="Eliminate or materialize an indicator whose premise is fixed by certified binary bounds.",
    ))
    reg.register(BridgeSpec(
        bridge_id="solverpilot.bridge.indicator.certified-big-m",
        bridge_version="1.0",
        source_kind="indicator",
        exactness=ExactnessClass.EXACT_EQUIVALENT,
        priority=20,
        description="Lower an affine indicator with finite box-bound-derived Big-M certificates.",
    ))
    return reg
