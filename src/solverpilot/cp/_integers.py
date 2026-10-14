"""Lossless integer normalization for CP inputs and candidate validation."""
from numbers import Integral, Real
import math


def integer(value, name="value"):
    if isinstance(value, Integral):
        return int(value)
    if isinstance(value, Real) and math.isfinite(value) and value == math.floor(value):
        return int(value)
    raise ValueError(f"{name} must be an exact finite integer")
