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
    from .workspace import _CURRENT
    call=_CURRENT.get()
    total, exponent = 0, 0
    count=0
    for (a, ae), (b, be) in pairs:
        count+=1
        if call is not None and count%256==0: call.tick(256)
        product = a*b
        if not product:
            continue
        power = ae+be
        if power > exponent:
            total <<= power-exponent
            exponent = power
        total += product << (exponent-power)
    if call is not None: call.tick(count%256)
    return Fraction(total, 1 << exponent)


def matvec(matrix, vector):
    matrix = matrix.tocsr()
    from .workspace import _CURRENT
    call=_CURRENT.get()
    if call is not None: call.tick(len(vector)+matrix.nnz)
    values = [_dyadic(v) for v in vector]
    prepared = None if call is None or call.workspace is None else call.workspace.prepare(matrix)
    if prepared is not None:
        pointers,indices,coefficients=prepared
        return [_sum_products((coefficients[k],values[indices[k]]) for k in range(pointers[i],pointers[i+1])) for i in range(matrix.shape[0])]
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
    from .workspace import _CURRENT
    call=_CURRENT.get()
    if call is not None: call.tick(len(coefficients))
    for coefficient, lo, hi in zip(coefficients, lower, upper):
        if not coefficient:
            continue
        bound = lo if coefficient > 0 else hi
        if not np.isfinite(bound):
            return None
        result += coefficient * rational(bound)
    return result
