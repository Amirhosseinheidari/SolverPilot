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


@dataclass(frozen=True, slots=True)
class ExponentialCone:
    """Closure of y exp(x/y) <= z, y > 0; vector ordering is (x,y,z)."""


@dataclass(frozen=True, slots=True)
class PowerCone:
    alpha: float

    def __post_init__(self):
        import math
        if isinstance(self.alpha, bool) or not math.isfinite(self.alpha) or not 0 < self.alpha < 1:
            raise ValueError('power exponent must be finite and strictly between zero and one')
        object.__setattr__(self, 'alpha', float(self.alpha))


ConstraintSet = ScalarSet | SecondOrderCone | RotatedSecondOrderCone | PositiveSemidefiniteCone | ExponentialCone | PowerCone
