from __future__ import annotations

from dataclasses import dataclass
from enum import Enum


@dataclass(frozen=True, slots=True, order=True)
class EntityId:
    value: str
    namespace: str = ""

    def __str__(self) -> str:
        return self.value

    @property
    def qualified(self) -> str:
        return f"{self.namespace}:{self.value}" if self.namespace else self.value


class SignDomain(str, Enum):
    UNKNOWN = "unknown"
    ZERO = "zero"
    NONNEGATIVE = "nonnegative"
    NONPOSITIVE = "nonpositive"


class Curvature(str, Enum):
    CONSTANT = "constant"
    AFFINE = "affine"
    UNKNOWN = "unknown"


class NumericType(str, Enum):
    REAL = "real"
    INTEGER = "integer"
    BINARY = "binary"
