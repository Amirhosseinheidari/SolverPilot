"""Execution identity is owned by results, never invented by a renderer."""

from functools import lru_cache
import hashlib
import json
from pathlib import Path
from uuid import uuid4


def new_execution_id() -> str:
    return str(uuid4())


def execution_id(result) -> str | None:
    """Return an existing ID; third-party objects without identity remain unknown."""
    value = getattr(result, "execution_id", None)
    if isinstance(value, str) and value.strip():
        return value
    trace = getattr(result, "trace", None)
    value = getattr(trace, "execution_id", None)
    if isinstance(value, str) and value.strip():
        return value
    for name in ("core_result", "result"):
        nested = getattr(result, name, None)
        if nested is not None and nested is not result:
            return execution_id(nested)
    return None


def source_tree_sha256() -> str:
    """Hash package Python source bytes, excluding local paths and model data."""
    root = Path(__file__).resolve().parent
    hashes = {
        p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(root.rglob("*.py"))
    }
    return hashlib.sha256(
        json.dumps(hashes, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


@lru_cache(maxsize=1)
def process_source_sha256() -> str | None:
    """Source snapshot at first traced execution, not binary/runtime attestation.

    Restart the interpreter after editing package source. Unreadable source is
    explicitly unavailable and must not prevent an otherwise supported solve.
    """
    try:
        return source_tree_sha256()
    except OSError:
        return None
