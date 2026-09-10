"""Strict scalar normalization for safe JSON/CSV mappings."""

from __future__ import annotations

import math
import re
from decimal import Decimal
from typing import Any

from .binary64 import MAX_SAFE_FLOAT_INTEGER, Binary64ConversionError, decimal_token_to_binary64
from .errors import DataIOMappingError

_INTEGER_RE = re.compile(r"^[+-]?(?:0|[1-9][0-9]*)$")
_NUMBER_RE = re.compile(
    r"^[+-]?(?:(?:0|[1-9][0-9]*)(?:\.[0-9]+)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?$"
)
_TRUE_VALUES = frozenset({"true", "1", "yes", "y"})
_FALSE_VALUES = frozenset({"false", "0", "no", "n"})
_MAX_INTEGER_DIGITS = 1000


def parse_integer(value: Any, *, field_name: str = "value") -> int:
    if isinstance(value, bool):
        raise DataIOMappingError(f"{field_name} must be an integer, not a boolean.")
    if isinstance(value, int):
        return value
    if not isinstance(value, str):
        raise DataIOMappingError(f"{field_name} must be an integer or integer string.")
    text = value.strip()
    if not _INTEGER_RE.fullmatch(text):
        raise DataIOMappingError(f"{field_name} is not a valid base-10 integer.")
    if len(text.lstrip("+-")) > _MAX_INTEGER_DIGITS:
        raise DataIOMappingError(f"{field_name} exceeds the supported integer digit limit.")
    try:
        return int(text, 10)
    except ValueError as exc:
        raise DataIOMappingError(f"{field_name} cannot be parsed as an integer safely.") from exc


def _parse_decimal_token(value: str, *, field_name: str, strict_binary64: bool) -> float:
    try:
        return decimal_token_to_binary64(value, exact_decimal=strict_binary64)
    except Binary64ConversionError as exc:
        reason = str(exc)
        if "exact binary64 range" in reason:
            raise DataIOMappingError(
                f"{field_name} is an integer outside exact IEEE-754 range; use integer data."
            ) from exc
        if "underflows" in reason:
            raise DataIOMappingError(f"{field_name} underflows IEEE-754 binary64.") from exc
        if "loses precision" in reason:
            raise DataIOMappingError(
                f"{field_name} loses decimal precision in IEEE-754 binary64 under strict policy."
            ) from exc
        raise DataIOMappingError(
            f"{field_name} cannot be represented safely as a finite IEEE-754 binary64 number."
        ) from exc


def parse_number(
    value: Any,
    *,
    field_name: str = "value",
    strict_binary64: bool = True,
) -> float:
    if type(strict_binary64) is not bool:
        raise DataIOMappingError("strict_binary64 must be a boolean.")
    if isinstance(value, bool):
        raise DataIOMappingError(f"{field_name} must be numeric, not a boolean.")
    if isinstance(value, int):
        return _parse_decimal_token(str(value), field_name=field_name, strict_binary64=strict_binary64)
    if isinstance(value, float):
        if not math.isfinite(value):
            raise DataIOMappingError(f"{field_name} must be finite.")
        if value.is_integer() and abs(value) > MAX_SAFE_FLOAT_INTEGER:
            raise DataIOMappingError(
                f"{field_name} is an integer outside exact IEEE-754 range; use integer data."
            )
        return value
    if isinstance(value, Decimal):
        if not value.is_finite():
            raise DataIOMappingError(f"{field_name} must be finite.")
        return _parse_decimal_token(
            str(value), field_name=field_name, strict_binary64=strict_binary64
        )
    if isinstance(value, str):
        text = value.strip()
        if not _NUMBER_RE.fullmatch(text):
            raise DataIOMappingError(f"{field_name} is not a valid decimal number.")
        return _parse_decimal_token(text, field_name=field_name, strict_binary64=strict_binary64)
    raise DataIOMappingError(f"{field_name} must be a number or numeric string.")


def parse_boolean(value: Any, *, field_name: str = "value") -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, int) and value in {0, 1}:
        return bool(value)
    if not isinstance(value, str):
        raise DataIOMappingError(f"{field_name} must be a boolean or boolean string.")
    normalized = value.strip().lower()
    if normalized in _TRUE_VALUES:
        return True
    if normalized in _FALSE_VALUES:
        return False
    raise DataIOMappingError(
        f"{field_name} is not a supported boolean value; use true/false, yes/no, or 1/0."
    )
