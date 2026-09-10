"""Deterministic decimal-to-binary64 conversion policies used by safe I/O."""

from __future__ import annotations

import math
from decimal import Decimal, InvalidOperation

MAX_SAFE_FLOAT_INTEGER = 2**53 - 1


class Binary64ConversionError(ValueError):
    """Raised when a decimal token violates the selected binary64 policy."""


def decimal_token_to_binary64(
    value: str,
    *,
    exact_decimal: bool = True,
    max_safe_integer: int = MAX_SAFE_FLOAT_INTEGER,
) -> float:
    """Convert a decimal token to finite IEEE-754 binary64.

    ``exact_decimal=True`` preserves the strict legacy connector invariant: the
    shortest decimal spelling of the resulting float must represent the same
    exact decimal value. This rejects silent ingestion changes such as
    ``0.10000000000000001 -> 0.1``. With ``exact_decimal=False`` only finite
    representability, non-underflow, and the exact integer boundary are enforced.
    """
    if not isinstance(value, str) or not value:
        raise Binary64ConversionError("decimal token must be a non-empty string")
    try:
        decimal_value = Decimal(value)
    except InvalidOperation as exc:
        raise Binary64ConversionError("invalid decimal token") from exc
    if not decimal_value.is_finite():
        raise Binary64ConversionError("non-finite decimal token")
    if decimal_value == decimal_value.to_integral_value() and abs(decimal_value) > max_safe_integer:
        raise Binary64ConversionError("integral decimal outside exact binary64 range")
    try:
        parsed = float(value)
    except (ValueError, OverflowError) as exc:
        raise Binary64ConversionError("decimal token cannot be represented as binary64") from exc
    if not math.isfinite(parsed):
        raise Binary64ConversionError("decimal token overflows binary64")
    if decimal_value != 0 and parsed == 0.0:
        raise Binary64ConversionError("decimal token underflows binary64")
    if exact_decimal and Decimal(str(parsed)) != decimal_value:
        raise Binary64ConversionError("decimal token loses precision in binary64")
    return parsed
