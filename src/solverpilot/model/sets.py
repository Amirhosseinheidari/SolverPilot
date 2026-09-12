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


@dataclass(frozen=True, slots=True)
class GeneralizedPowerCone:
    """prod(x[i]**weights[i]) >= norm(y), with nonnegative x.

    Ordering is (x, y); weights sum to one and y has tail_dimension entries.
    Weights within 1e-12 of unit sum are normalized once in this immutable set.
    """
    weights: tuple[float, ...]
    tail_dimension: int = 1

    def __post_init__(self):
        import math
        try:
            raw = tuple(self.weights)
            weights = tuple(float(v) for v in raw)
        except (TypeError, ValueError) as exc:
            raise ValueError("weights must be a finite positive sequence") from exc
        if len(weights) < 2 or any(isinstance(v, bool) for v in raw) or any(not math.isfinite(v) or v <= 0 for v in weights):
            raise ValueError("at least two finite positive weights are required")
        total = math.fsum(weights)
        if not math.isclose(total, 1.0, rel_tol=0., abs_tol=1e-12):
            raise ValueError("generalized power weights must sum to one")
        if type(self.tail_dimension) is not int or self.tail_dimension < 1:
            raise ValueError("tail_dimension must be a positive integer")
        object.__setattr__(self, "weights", tuple(v / total for v in weights))

    @property
    def dimension(self):
        return len(self.weights) + self.tail_dimension


ConstraintSet = ScalarSet | SecondOrderCone | RotatedSecondOrderCone | PositiveSemidefiniteCone | ExponentialCone | PowerCone | GeneralizedPowerCone
