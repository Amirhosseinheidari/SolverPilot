from __future__ import annotations

from enum import Enum


class ObjectiveSense(str, Enum):
    MINIMIZE = "minimize"
    MAXIMIZE = "maximize"


class VariableDomain(str, Enum):
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    BINARY = "binary"


class ConvexityStatus(str, Enum):
    CONFIRMED = "confirmed"
    REJECTED = "rejected"
    UNKNOWN = "unknown"
