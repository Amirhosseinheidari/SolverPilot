"""Immutable result object returned by SolverPilot data loaders."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Any

from .errors import DataIOConfigurationError
from .hashing import deep_freeze_json, deep_thaw_json
from .model import DataIssue
from .provenance import SourceProvenance


@dataclass(frozen=True, slots=True)
class LoadedData:
    """Normalized immutable payload plus provenance and non-fatal issues."""

    payload: Any
    provenance: SourceProvenance
    issues: tuple[DataIssue, ...] = ()
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not isinstance(self.provenance, SourceProvenance):
            raise DataIOConfigurationError("provenance must be SourceProvenance.")
        issues = tuple(self.issues)
        if any(not isinstance(issue, DataIssue) for issue in issues):
            raise DataIOConfigurationError("issues must contain DataIssue values.")
        frozen_metadata = deep_freeze_json(self.metadata)
        if not isinstance(frozen_metadata, Mapping):
            raise DataIOConfigurationError("metadata must form an object.")
        object.__setattr__(self, "payload", deep_freeze_json(self.payload))
        object.__setattr__(self, "issues", issues)
        object.__setattr__(self, "metadata", frozen_metadata)

    def mutable_payload(self) -> Any:
        return deep_thaw_json(self.payload)

    def evidence_dict(self) -> dict[str, Any]:
        return {
            "provenance": self.provenance.to_dict(),
            "issues": [issue.to_dict() for issue in self.issues],
            "metadata": deep_thaw_json(self.metadata),
        }
