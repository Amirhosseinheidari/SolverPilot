"""Explicitly refreshable cache of installed distribution versions.

Only metadata is shared; mutable solver instances are never process-global.
"""
from functools import lru_cache
from importlib.metadata import version as _installed_version


@lru_cache(maxsize=128)
def version(distribution: str) -> str:
    return _installed_version(distribution)


def refresh_backend_metadata() -> None:
    """Forget version metadata after changing packages in a running interpreter."""
    version.cache_clear()
