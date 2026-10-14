from __future__ import annotations

from collections.abc import Mapping
from types import MappingProxyType
from typing import Any

import numpy as np


class _CanonicalArray(np.ndarray):
    """Immutable stored data and metadata; derived copies remain ordinary arrays."""

    def __setattr__(self, name, value):
        if name in {'shape', 'dtype', 'strides', 'data', '__class__'}:
            raise ValueError('canonical array metadata is immutable')
        super().__setattr__(name, value)

    def resize(self, *args, **kwargs):
        raise ValueError('canonical array metadata is immutable')

    def copy(self, order='C'):
        return np.array(self, copy=True, subok=False, order=order)

    def __reduce__(self):
        return canonical_array, (np.asarray(self),)

    def __reduce_ex__(self, protocol):
        return self.__reduce__()


def canonical_array(value, dtype=None):
    """Own numeric storage whose values, shape and dtype cannot be changed."""
    array = readonly_array(value, dtype=dtype)
    if array.dtype.hasobject:
        raise TypeError('canonical arrays require a non-object dtype')
    return array.view(_CanonicalArray)


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


def deep_thaw(value: Any) -> Any:
    """Make immutable mappings pickleable without changing numeric contents."""
    if isinstance(value, Mapping):
        return {k: deep_thaw(v) for k, v in value.items()}
    if isinstance(value, tuple):
        return tuple(deep_thaw(v) for v in value)
    if isinstance(value, frozenset):
        return frozenset(deep_thaw(v) for v in value)
    return value
