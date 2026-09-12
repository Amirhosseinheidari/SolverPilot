from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


def readonly_array(value: np.ndarray | Any, *, dtype=None) -> np.ndarray:
    """Return an independent read-only snapshot, with immutable numeric storage."""
    arr = np.asarray(value, dtype=dtype)
    if arr.dtype.hasobject:
        # Object pointers cannot safely be reconstructed from a byte buffer.
        arr = arr.copy()
        arr.flags.writeable = False
        return arr
    return np.frombuffer(arr.tobytes(order="C"), dtype=arr.dtype).reshape(arr.shape)


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
