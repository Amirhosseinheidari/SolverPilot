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
    return _sum_products((_dyadic(a), _dyadic(b)) for a, b in zip(left, right))


def _dyadic(value):
    """Exact integer numerator and power-of-two denominator of binary64."""
    numerator, denominator = float(value).as_integer_ratio()
    return numerator, denominator.bit_length()-1


def _sum_products(pairs):
    # Align integer products, reducing to Fraction only once per dot product.
    # No floating-point multiply/add or discarded low bits are involved.
    total, exponent = 0, 0
    for (a, ae), (b, be) in pairs:
        product = a*b
        if not product:
            continue
        power = ae+be
        if power > exponent:
            total <<= power-exponent
            exponent = power
        total += product << (exponent-power)
    return Fraction(total, 1 << exponent)


def matvec(matrix, vector):
    matrix = matrix.tocsr()
    values = [_dyadic(v) for v in vector]
    return [
        _sum_products(
            (
                (_dyadic(matrix.data[k]), values[matrix.indices[k]])
                for k in range(matrix.indptr[i], matrix.indptr[i + 1])
            ),
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
