from __future__ import annotations

import hashlib
from typing import Iterable

import numpy as np
from scipy import sparse


def _hash_parts(parts: Iterable[bytes]) -> str:
    h = hashlib.sha256()
    for part in parts:
        h.update(len(part).to_bytes(8, byteorder="little", signed=False))
        h.update(part)
    return h.hexdigest()


def _array_bytes(arr: np.ndarray) -> bytes:
    c = np.ascontiguousarray(arr)
    return b"|".join(
        [
            str(c.dtype).encode(),
            repr(tuple(c.shape)).encode(),
            c.tobytes(order="C"),
        ]
    )


def hash_array(arr: np.ndarray) -> str:
    return _hash_parts([_array_bytes(arr)])


def hash_sparse_matrix(matrix: sparse.spmatrix) -> str:
    csr = sparse.csr_matrix(matrix, dtype=np.float64, copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    return _hash_parts(
        [
            repr(csr.shape).encode(),
            _array_bytes(csr.indptr.astype(np.int64, copy=False)),
            _array_bytes(csr.indices.astype(np.int64, copy=False)),
            _array_bytes(csr.data.astype(np.float64, copy=False)),
        ]
    )


def structural_hash_linear(
    A: sparse.spmatrix,
    domains: np.ndarray,
) -> str:
    csr = sparse.csr_matrix(A, dtype=np.float64, copy=True)
    csr.sum_duplicates()
    csr.sort_indices()
    # Structural hash intentionally excludes numerical matrix values.
    return _hash_parts(
        [
            repr(csr.shape).encode(),
            _array_bytes(csr.indptr.astype(np.int64, copy=False)),
            _array_bytes(csr.indices.astype(np.int64, copy=False)),
            _array_bytes(np.asarray(domains, dtype="U16")),
        ]
    )


def data_hash_linear(
    A: sparse.spmatrix,
    c: np.ndarray,
    variable_lower: np.ndarray,
    variable_upper: np.ndarray,
    constraint_lower: np.ndarray,
    constraint_upper: np.ndarray,
    domains: np.ndarray,
    objective_sense: str,
    objective_offset: float,
) -> str:
    return _hash_parts(
        [
            hash_sparse_matrix(A).encode(),
            _array_bytes(c),
            _array_bytes(variable_lower),
            _array_bytes(variable_upper),
            _array_bytes(constraint_lower),
            _array_bytes(constraint_upper),
            _array_bytes(np.asarray(domains, dtype="U16")),
            objective_sense.encode(),
            np.float64(objective_offset).tobytes(),
        ]
    )
