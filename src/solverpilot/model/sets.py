from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class LessThan:
    upper: float = 0.0


@dataclass(frozen=True, slots=True)
class GreaterThan:
    lower: float = 0.0


@dataclass(frozen=True, slots=True)
class EqualTo:
    value: float = 0.0


@dataclass(frozen=True, slots=True)
class Interval:
    lower: float
    upper: float


ScalarSet = LessThan | GreaterThan | EqualTo | Interval


@dataclass(frozen=True, slots=True)
class SecondOrderCone:
    dimension: int

    def __post_init__(self) -> None:
        if int(self.dimension) < 2:
            raise ValueError("SecondOrderCone dimension must be >= 2")
        object.__setattr__(self, "dimension", int(self.dimension))


@dataclass(frozen=True, slots=True)
class RotatedSecondOrderCone:
    dimension: int

    def __post_init__(self) -> None:
        if int(self.dimension) < 3:
            raise ValueError("RotatedSecondOrderCone dimension must be >= 3")
        object.__setattr__(self, "dimension", int(self.dimension))


@dataclass(frozen=True, slots=True)
class PositiveSemidefiniteCone:
    dimension: int

    def __post_init__(self) -> None:
        if int(self.dimension) < 1:
            raise ValueError("PositiveSemidefiniteCone dimension must be >= 1")
        object.__setattr__(self, "dimension", int(self.dimension))


ConstraintSet = ScalarSet | SecondOrderCone | RotatedSecondOrderCone | PositiveSemidefiniteCone
