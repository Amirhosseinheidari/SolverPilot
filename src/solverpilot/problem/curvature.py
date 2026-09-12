"""Conservative PSD evidence about the supplied binary64 matrix.

Numerical eigensolver tolerances are not proofs of nonnegative curvature.
Sparse diagonal dominance and small rational LDL checks provide exact evidence;
an exhausted size/arithmetic budget returns unknown rather than accepting.
"""

from dataclasses import dataclass
from fractions import Fraction
import numpy as np
from scipy import sparse
from .hashing import hash_sparse_matrix


@dataclass(frozen=True, slots=True)
class PSDEvidence:
    status: str
    method: str
    matrix_hash: str
    reason: str

    @property
    def certified(self) -> bool:
        return self.status == "positive_semidefinite"


def certify_psd(
    matrix, *, max_dense_dimension: int = 32, max_fraction_bits: int = 8192
) -> PSDEvidence:
    """Prove PSD, prove a negative direction, or return unknown.

    This checks the exact floating-point input, not an intended unrounded model.
    It handles singular PSD matrices. Size/bit limits bound the dense fallback.
    """
    if type(max_dense_dimension) is not int or max_dense_dimension < 0:
        raise ValueError("max_dense_dimension must be a nonnegative integer")
    if type(max_fraction_bits) is not int or max_fraction_bits < 64:
        raise ValueError("max_fraction_bits must be an integer >= 64")
    a = sparse.csr_matrix(matrix, dtype=float, copy=True)
    a.sum_duplicates()
    a.eliminate_zeros()
    a.sort_indices()
    if a.shape[0] != a.shape[1] or not np.isfinite(a.data).all():
        raise ValueError("PSD evidence requires a finite square matrix")
    digest = hash_sparse_matrix(a)

    def result(status, method, reason):
        return PSDEvidence(status, method, digest, reason)

    if (a != a.T).nnz:
        return result("unknown", "symmetry", "matrix is not exactly symmetric")
    diagonal = a.diagonal()
    if np.any(diagonal < 0):
        return result("not_psd", "negative_diagonal", "coordinate direction has negative curvature")
    # A zero diagonal with a nonzero off-diagonal gives a negative 2x2 minor.
    for i in np.flatnonzero(diagonal == 0):
        if a.indptr[i + 1] > a.indptr[i]:
            return result("not_psd", "zero_diagonal_minor", "nonzero row at a zero diagonal")
    dominant = True
    for i in range(a.shape[0]):
        radius = sum(
            (
                abs(Fraction(float(a.data[k])))
                for k in range(a.indptr[i], a.indptr[i + 1])
                if a.indices[k] != i
            ),
            Fraction(0),
        )
        if Fraction(float(diagonal[i])) < radius:
            dominant = False
            break
    if dominant:
        return result(
            "positive_semidefinite",
            "exact_gershgorin",
            "all exact Gershgorin lower bounds are nonnegative",
        )
    if a.shape[0] > max_dense_dimension:
        return result("unknown", "size_limit", "matrix exceeds the rational LDL dimension budget")
    values = [[Fraction(float(v)) for v in row] for row in a.toarray()]
    n = len(values)
    for k in range(n):
        pivot = values[k][k]
        if pivot < 0 or (pivot == 0 and any(values[i][k] for i in range(k + 1, n))):
            return result("not_psd", "exact_ldl", "negative Schur-complement direction")
        if pivot == 0:
            continue
        for i in range(k + 1, n):
            for j in range(i, n):
                v = values[i][j] - values[i][k] * values[j][k] / pivot
                if max(v.numerator.bit_length(), v.denominator.bit_length()) > max_fraction_bits:
                    return result(
                        "unknown", "arithmetic_limit", "rational LDL exceeded its bit budget"
                    )
                values[i][j] = values[j][i] = v
    return result(
        "positive_semidefinite",
        "exact_ldl",
        "exact nonnegative LDL pivots including singular blocks",
    )
