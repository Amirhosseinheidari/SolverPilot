from __future__ import annotations

from dataclasses import dataclass, field

from .enums import Capability, SupportLevel


@dataclass(frozen=True, slots=True)
class BackendManifest:
    name: str
    version: str | None = None
    capabilities: dict[Capability, SupportLevel] = field(default_factory=dict)
    metadata: dict[str, object] = field(default_factory=dict)

    def support(self, capability: Capability) -> SupportLevel:
        return self.capabilities.get(capability, SupportLevel.UNKNOWN)

    def supports_without_risky_emulation(self, capability: Capability) -> bool:
        return self.support(capability) in {SupportLevel.NATIVE, SupportLevel.EMULATED_SAFE}
