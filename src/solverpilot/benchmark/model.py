from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping


_PROVENANCE_LEVELS = {
    "UNSPECIFIED",
    "VERIFIED_FROZEN",
    "PINNED_EXTERNAL",
    "PINNED_ARCHIVE_ONLY",
    "UNPINNED_EXTERNAL",
    "LOCAL_ONLY",
}


class _FrozenDict(dict):
    """Immutable dict compatible with dataclasses.asdict/json workflows."""

    def _blocked(self, *args, **kwargs):
        raise TypeError("immutable mapping")

    __setitem__ = __delitem__ = clear = pop = popitem = setdefault = update = _blocked

    def __deepcopy__(self, memo):
        return dict(self)


def _freeze(value: Any) -> Any:
    if isinstance(value, Mapping):
        return _FrozenDict({k: _freeze(v) for k, v in value.items()})
    if isinstance(value, list):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, tuple):
        return tuple(_freeze(v) for v in value)
    if isinstance(value, set):
        return frozenset(_freeze(v) for v in value)
    return value


@dataclass(frozen=True, slots=True)
class DatasetSpec:
    name: str
    kind: str
    archive_url: str | None = None
    manifest_url: str | None = None
    reference_url: str | None = None
    archive_sha256: str | None = None
    manifest_sha256: str | None = None
    reference_sha256: str | None = None
    archive_filename: str | None = None
    manifest_filename: str | None = None
    reference_filename: str | None = None
    instance_suffixes: tuple[str, ...] = (".mps.gz", ".mps")
    provenance_level: str = "UNSPECIFIED"
    metadata: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        level = str(self.provenance_level).strip().upper()
        if level not in _PROVENANCE_LEVELS:
            raise ValueError(f"unsupported provenance_level: {self.provenance_level!r}")
        object.__setattr__(self, "provenance_level", level)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        object.__setattr__(self, "instance_suffixes", tuple(self.instance_suffixes))


@dataclass(frozen=True, slots=True)
class RunSpec:
    dataset_dir: Path
    manifest: Path
    reference: Path | None
    backends: tuple[str, ...]
    output_jsonl: Path
    repetitions: int = 1
    time_limit_s: float | None = None
    hard_timeout_s: float | None = None
    objective_atol: float = 1e-6
    objective_rtol: float = 1e-7
    shard_index: int = 0
    shard_count: int = 1
    resume: bool = True
    seed: int = 0
    thread_env_limit: int | None = 1
    solver_threads: int | None = None
    worker_python_mode: str = "no_site"

    def __post_init__(self) -> None:
        if self.repetitions < 1:
            raise ValueError("repetitions must be >= 1")
        if self.time_limit_s is not None and self.time_limit_s <= 0:
            raise ValueError("time_limit_s must be positive")
        if self.hard_timeout_s is not None and self.hard_timeout_s <= 0:
            raise ValueError("hard_timeout_s must be positive")
        if self.thread_env_limit is not None and self.thread_env_limit < 1:
            raise ValueError("thread_env_limit must be >= 1")
        if self.solver_threads is not None and self.solver_threads < 1:
            raise ValueError("solver_threads must be >= 1")
        if self.worker_python_mode not in {"normal", "no_site"}:
            raise ValueError("worker_python_mode must be 'normal' or 'no_site'")
        if not self.backends:
            raise ValueError("backends must not be empty")
        if self.shard_count < 1:
            raise ValueError("shard_count must be >= 1")
        if not 0 <= self.shard_index < self.shard_count:
            raise ValueError("shard_index must be in [0, shard_count)")


@dataclass(frozen=True, slots=True)
class AcquisitionFile:
    role: str
    url: str
    path: str
    size_bytes: int
    sha256: str
    expected_sha256: str | None
    verified: bool


@dataclass(frozen=True, slots=True)
class AcquisitionRecord:
    dataset: str
    spec_kind: str
    target_dir: str
    files: tuple[AcquisitionFile, ...]
    extracted: bool
    extracted_files: int
    metadata: Mapping[str, Any]
    schema_version: str
    spec_fingerprint: str
    provenance_level: str
    all_remote_inputs_pinned: bool
    unpinned_roles: tuple[str, ...]

    def __post_init__(self) -> None:
        if not self.schema_version:
            raise ValueError("schema_version must not be empty")
        if len(self.spec_fingerprint) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in self.spec_fingerprint):
            raise ValueError("spec_fingerprint must be a SHA-256 hex digest")
        level = str(self.provenance_level).strip().upper()
        if level not in _PROVENANCE_LEVELS:
            raise ValueError(f"unsupported provenance_level: {self.provenance_level!r}")
        object.__setattr__(self, "provenance_level", level)
        object.__setattr__(self, "metadata", _freeze(dict(self.metadata)))
        object.__setattr__(self, "files", tuple(self.files))
        object.__setattr__(self, "unpinned_roles", tuple(sorted(set(self.unpinned_roles))))
