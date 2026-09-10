from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def readonly_array(value: np.ndarray | Any, *, dtype=None) -> np.ndarray:
    """Return an owned, read-only NumPy snapshot."""
    arr = np.array(value, dtype=dtype, copy=True)
    arr.flags.writeable = False
    return arr


def deep_freeze(value: Any) -> Any:
    """Recursively snapshot common mutable containers into immutable forms."""
    if isinstance(value, np.ndarray):
        return readonly_array(value)
    if isinstance(value, Mapping):
        return MappingProxyType({k: deep_freeze(v) for k, v in value.items()})
    if isinstance(value, (list, tuple)):
        return tuple(deep_freeze(v) for v in value)
    if isinstance(value, (set, frozenset)):
        return frozenset(deep_freeze(v) for v in value)
    return value
