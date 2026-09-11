"""Exact operations on the represented binary64 data for certificate bounds.

These operations certify the supplied floating-point model, not an unknown
pre-rounding real model. Sparse products visit only stored nonzero entries.
"""

from fractions import Fraction
import numpy as np


def rational(value):
    return Fraction(float(value))


def bounded_float(value):
    try:
        return float(value)
    except OverflowError:
        return np.inf if value > 0 else -np.inf


def dot(left, right):
    return sum((rational(a) * rational(b) for a, b in zip(left, right)), Fraction())


def matvec(matrix, vector):
    matrix = matrix.tocsr()
    values = [rational(v) for v in vector]
    return [
        sum(
            (
                rational(matrix.data[k]) * values[matrix.indices[k]]
                for k in range(matrix.indptr[i], matrix.indptr[i + 1])
            ),
            Fraction(),
        )
        for i in range(matrix.shape[0])
    ]


def box_min(coefficients, lower, upper):
    """Return an exact support bound, or None for an unbounded direction."""
    result = Fraction()
    for coefficient, lo, hi in zip(coefficients, lower, upper):
        if not coefficient:
            continue
        bound = lo if coefficient > 0 else hi
        if not np.isfinite(bound):
            return None
        result += coefficient * rational(bound)
    return result
