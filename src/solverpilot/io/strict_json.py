"""Strict JSON parser shared by file ingestion and JSON-valued mapping fields."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from typing import Any

from .binary64 import Binary64ConversionError, decimal_token_to_binary64
from .errors import DataIOParseError

DEFAULT_MAX_JSON_DEPTH = 256
DEFAULT_MAX_INTEGER_DIGITS = 1000


class _DuplicateKey(ValueError):
    pass


class _NonFiniteConstant(ValueError):
    pass


class _IntegerTooLarge(ValueError):
    pass


def _object_pairs_no_duplicates(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
    result: dict[str, Any] = {}
    for key, value in pairs:
        if key in result:
            raise _DuplicateKey("duplicate object key")
        result[key] = value
    return result


def _reject_constant(value: str) -> None:
    raise _NonFiniteConstant(value)


def _parse_bounded_int(value: str) -> int:
    if len(value.lstrip("-")) > DEFAULT_MAX_INTEGER_DIGITS:
        raise _IntegerTooLarge("integer digit limit exceeded")
    return int(value, 10)


def _validate_tree(value: Any, *, max_depth: int, context: str) -> None:
    stack: list[tuple[Any, int]] = [(value, 0)]
    while stack:
        current, depth = stack.pop()
        if depth > max_depth:
            raise DataIOParseError(f"{context} exceeds the maximum nesting depth of {max_depth}.")
        if current is None or isinstance(current, (bool, int)):
            continue
        if isinstance(current, float):
            if not math.isfinite(current):
                raise DataIOParseError(f"{context} contains a non-finite number.")
            continue
        if isinstance(current, str):
            try:
                current.encode("utf-8")
            except UnicodeEncodeError as exc:
                raise DataIOParseError(f"{context} contains an invalid Unicode surrogate.") from exc
            continue
        if isinstance(current, Mapping):
            for key, item in current.items():
                if not isinstance(key, str):
                    raise DataIOParseError(f"{context} object keys must be strings.")
                try:
                    key.encode("utf-8")
                except UnicodeEncodeError as exc:
                    raise DataIOParseError(
                        f"{context} contains an invalid Unicode surrogate in an object key."
                    ) from exc
                stack.append((item, depth + 1))
            continue
        if isinstance(current, Sequence) and not isinstance(current, (str, bytes, bytearray)):
            stack.extend((item, depth + 1) for item in current)
            continue
        raise DataIOParseError(f"{context} contains an unsupported parsed value type.")


def loads_strict_json(
    text: str,
    *,
    context: str = "JSON input",
    max_depth: int = DEFAULT_MAX_JSON_DEPTH,
    strict_numbers: bool = True,
) -> Any:
    """Parse standards-compliant JSON with deterministic resource and numeric gates."""
    if not isinstance(text, str):
        raise DataIOParseError(f"{context} must be text.")
    if type(max_depth) is not int or max_depth < 0:
        raise DataIOParseError("max_depth must be a non-negative integer.")
    if type(strict_numbers) is not bool:
        raise DataIOParseError("strict_numbers must be a boolean.")

    def parse_float(token: str) -> float:
        try:
            return decimal_token_to_binary64(token, exact_decimal=strict_numbers)
        except Binary64ConversionError as exc:
            raise Binary64ConversionError(str(exc)) from exc

    try:
        parsed = json.loads(
            text,
            object_pairs_hook=_object_pairs_no_duplicates,
            parse_constant=_reject_constant,
            parse_int=_parse_bounded_int,
            parse_float=parse_float,
        )
    except _DuplicateKey as exc:
        raise DataIOParseError(f"{context} contains a duplicate object key.") from exc
    except _NonFiniteConstant as exc:
        raise DataIOParseError(f"{context} contains a non-finite JSON constant.") from exc
    except _IntegerTooLarge as exc:
        raise DataIOParseError(
            f"{context} contains an integer exceeding {DEFAULT_MAX_INTEGER_DIGITS} digits."
        ) from exc
    except Binary64ConversionError as exc:
        raise DataIOParseError(
            f"{context} contains a decimal number rejected by the binary64 ingestion policy."
        ) from exc
    except json.JSONDecodeError as exc:
        raise DataIOParseError(f"{context} is malformed at line {exc.lineno}, column {exc.colno}.") from exc
    except RecursionError as exc:
        raise DataIOParseError(f"{context} exceeds the supported nesting depth.") from exc
    except (ValueError, OverflowError, UnicodeError) as exc:
        raise DataIOParseError(f"{context} cannot be parsed safely.") from exc

    _validate_tree(parsed, max_depth=max_depth, context=context)
    return parsed
