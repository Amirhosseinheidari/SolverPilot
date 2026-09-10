"""Fail-closed helpers for local file and declared-bundle ingestion."""

from __future__ import annotations

import os
from pathlib import Path, PurePosixPath

from .errors import DataIOSourceError


def _absolute_without_resolving(path: Path) -> Path:
    return Path(os.path.abspath(os.fspath(path)))


def _reject_symlink_components(path: Path) -> None:
    absolute = _absolute_without_resolving(path)
    current = Path(absolute.anchor)
    for part in absolute.parts[1:]:
        current = current / part
        try:
            if current.is_symlink():
                raise DataIOSourceError("Symbolic links are rejected in data source paths.")
        except OSError as exc:
            raise DataIOSourceError("Local source path components cannot be inspected safely.") from exc


def validate_local_path(
    path: str | Path,
    *,
    expect_directory: bool | None,
    reject_symlinks: bool = True,
) -> Path:
    if not isinstance(path, (str, os.PathLike)):
        raise DataIOSourceError("Local source path must be path-like.")
    if isinstance(path, str) and not path.strip():
        raise DataIOSourceError("Local source path must be non-empty.")
    candidate = Path(path)
    if reject_symlinks:
        _reject_symlink_components(candidate)
    try:
        exists = candidate.exists()
    except OSError as exc:
        raise DataIOSourceError("Local source path cannot be inspected safely.") from exc
    if not exists:
        raise DataIOSourceError(f"Local source does not exist: {candidate.name or 'source'}.")
    try:
        is_directory = candidate.is_dir()
        is_file = candidate.is_file()
    except OSError as exc:
        raise DataIOSourceError("Local source type cannot be inspected safely.") from exc
    if expect_directory is True and not is_directory:
        raise DataIOSourceError("Data source must be a directory bundle.")
    if expect_directory is False and not is_file:
        raise DataIOSourceError("Data source must be a regular file.")
    if expect_directory is None and not (is_file or is_directory):
        raise DataIOSourceError("Data source must be a regular file or directory bundle.")
    return candidate


def safe_bundle_child(
    bundle: Path,
    file_name: str,
    *,
    reject_symlinks: bool = True,
) -> Path:
    if not isinstance(file_name, str) or not file_name.strip():
        raise DataIOSourceError("Bundle file name must be a non-empty string.")
    posix = PurePosixPath(file_name.strip().replace("\\", "/"))
    if posix.is_absolute() or not posix.parts or any(
        part in {"", ".", ".."} or ":" in part for part in posix.parts
    ):
        raise DataIOSourceError("Bundle file name must be a safe relative path.")
    child = bundle.joinpath(*posix.parts)
    if reject_symlinks:
        _reject_symlink_components(bundle)
        _reject_symlink_components(child)
    try:
        bundle_resolved = bundle.resolve(strict=True)
        child_resolved = child.resolve(strict=True)
    except FileNotFoundError as exc:
        raise DataIOSourceError(f"Declared bundle file is missing: {posix.as_posix()}.") from exc
    except OSError as exc:
        raise DataIOSourceError("Bundle paths cannot be resolved safely.") from exc
    if bundle_resolved != child_resolved.parent and bundle_resolved not in child_resolved.parents:
        raise DataIOSourceError("Declared bundle file escapes the bundle directory.")
    if not child_resolved.is_file():
        raise DataIOSourceError(f"Declared bundle entry is not a file: {posix.as_posix()}.")
    return child_resolved


def read_limited_bytes(path: Path, *, max_bytes: int) -> bytes:
    """Read at most ``max_bytes`` and detect growth without trusting a prior stat."""
    if type(max_bytes) is not int or max_bytes <= 0:
        raise DataIOSourceError("Maximum source size must be a positive integer.")
    try:
        with path.open("rb") as handle:
            data = handle.read(max_bytes + 1)
    except OSError as exc:
        raise DataIOSourceError("Source bytes could not be read.") from exc
    if len(data) > max_bytes:
        raise DataIOSourceError(
            f"Source exceeds the configured maximum size of {max_bytes} bytes."
        )
    return data
