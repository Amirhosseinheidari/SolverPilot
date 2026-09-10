from __future__ import annotations

from dataclasses import dataclass, asdict
import json
from pathlib import Path

from solverpilot.evaluation import parse_miplib_solu, parse_test_manifest
from .acquire import sha256_file

_SCIENTIFIC_PROVENANCE_LEVELS = {"VERIFIED_FROZEN", "PINNED_EXTERNAL"}


def _valid_sha256(value: object) -> bool:
    text = str(value or "")
    return len(text) == 64 and all(ch in "0123456789abcdefABCDEF" for ch in text)


@dataclass(frozen=True, slots=True)
class DatasetVerification:
    ok: bool
    target_dir: str
    file_hashes_ok: bool
    lock_integrity_ok: bool
    manifest_entries: int | None
    reference_records: int | None
    instances_present: int | None
    instances_missing: tuple[str, ...]
    provenance_level: str
    all_remote_inputs_pinned: bool
    unpinned_roles: tuple[str, ...]
    scientific_provenance_verified: bool
    errors: tuple[str, ...]

    def to_dict(self) -> dict:
        return asdict(self)


def verify_acquired_dataset(target_dir: str | Path) -> DatasetVerification:
    target = Path(target_dir)
    lock_path = target / "acquisition-lock.json"
    if not lock_path.exists():
        raise FileNotFoundError(lock_path)

    errors: list[str] = []
    lock_integrity_ok = True
    lock_sha_path = target / "acquisition-lock.sha256"
    if lock_sha_path.exists():
        parts = lock_sha_path.read_text(encoding="utf-8").strip().split()
        expected_lock_sha = parts[0] if parts else ""
        if not _valid_sha256(expected_lock_sha) or sha256_file(lock_path).lower() != expected_lock_sha.lower():
            lock_integrity_ok = False
            errors.append("acquisition lock checksum mismatch")

    lock = json.loads(lock_path.read_text(encoding="utf-8"))
    lock_schema_version = str(lock.get("schema_version") or "legacy")
    if lock_schema_version == "2.0" and not lock_sha_path.exists():
        lock_integrity_ok = False
        errors.append("schema 2.0 acquisition lock is missing acquisition-lock.sha256")
    if lock_schema_version == "2.0" and not _valid_sha256(lock.get("spec_fingerprint")):
        errors.append("schema 2.0 acquisition lock has invalid spec_fingerprint")

    provenance_level = str(lock.get("provenance_level") or "UNSPECIFIED").strip().upper()
    hashes_ok = True
    role_paths: dict[str, Path] = {}
    observed_unpinned: list[str] = []
    files = lock.get("files", [])
    if not isinstance(files, list):
        raise ValueError("acquisition lock files must be a list")

    for item in files:
        if not isinstance(item, dict):
            hashes_ok = False; errors.append("invalid acquisition file record"); continue
        role = str(item.get("role") or "")
        path = Path(str(item.get("path") or ""))
        if role: role_paths[role] = path
        expected = item.get("expected_sha256")
        if expected is None:
            if role: observed_unpinned.append(role)
        elif not _valid_sha256(expected):
            hashes_ok = False; errors.append(f"invalid expected SHA-256 for role {role!r}")
        if not path.exists():
            hashes_ok = False; errors.append(f"missing acquired file: {path}"); continue
        observed = sha256_file(path); recorded = item.get("sha256")
        if not _valid_sha256(recorded) or observed.lower() != str(recorded).lower():
            hashes_ok = False; errors.append(f"hash mismatch: {path}")
        if expected is not None and observed.lower() != str(expected).lower():
            hashes_ok = False; errors.append(f"expected hash mismatch: {path}")
        if expected is not None and item.get("verified") is not True:
            hashes_ok = False; errors.append(f"pinned role {role!r} was not recorded as verified")
        if expected is None and item.get("verified") is True:
            hashes_ok = False; errors.append(f"unpinned role {role!r} cannot be recorded as verified")

    observed_unpinned_tuple = tuple(sorted(set(observed_unpinned)))
    computed_all_pinned = bool(files) and not observed_unpinned_tuple
    if lock_schema_version == "2.0":
        recorded_unpinned = tuple(sorted(set(str(x) for x in lock.get("unpinned_roles", ()))))
        recorded_all_pinned = bool(lock.get("all_remote_inputs_pinned", False))
        if recorded_unpinned != observed_unpinned_tuple:
            errors.append(f"acquisition provenance mismatch: recorded unpinned_roles={recorded_unpinned}, observed={observed_unpinned_tuple}")
        if recorded_all_pinned != computed_all_pinned:
            errors.append("acquisition provenance mismatch: all_remote_inputs_pinned does not match file records")

    manifest_entries = reference_records = present = None
    missing: list[str] = []
    manifest = role_paths.get("manifest")
    if manifest is not None and manifest.exists():
        try:
            filenames = parse_test_manifest(manifest.read_text(encoding="utf-8")); manifest_entries = len(filenames)
            expected = lock.get("metadata", {}).get("benchmark_instances")
            if expected is not None and manifest_entries != int(expected): errors.append(f"manifest count {manifest_entries} != expected {expected}")
            instance_root = target / "instances"
            if instance_root.exists():
                for filename in filenames:
                    direct = instance_root / filename
                    if direct.exists(): continue
                    matches = list(instance_root.rglob(filename))
                    if len(matches) != 1: missing.append(filename)
                present = manifest_entries - len(missing)
        except Exception as exc:
            errors.append(f"manifest verification failed: {type(exc).__name__}: {exc}")
    if manifest is None and str(lock.get("spec_kind")) == "qplib":
        instance_root = target / "instances"
        if instance_root.exists():
            present = len(list(instance_root.rglob("*.qplib")))
            metadata = lock.get("metadata", {})
            if metadata.get("continuous_instances") is not None and metadata.get("discrete_instances") is not None:
                expected = int(metadata["continuous_instances"]) + int(metadata["discrete_instances"])
                if present != expected: errors.append(f"QPLIB extracted count {present} != expected {expected}")
    reference = role_paths.get("reference")
    if reference is not None and reference.exists():
        try:
            refs = parse_miplib_solu(reference.read_text(encoding="utf-8")); reference_records = len(refs)
            expected = lock.get("metadata", {}).get("solution_records")
            if expected is not None and reference_records != int(expected): errors.append(f"reference record count {reference_records} != expected {expected}")
        except Exception as exc:
            errors.append(f"reference verification failed: {type(exc).__name__}: {exc}")

    local_ok = lock_integrity_ok and hashes_ok and not errors and not missing
    scientific = local_ok and computed_all_pinned and provenance_level in _SCIENTIFIC_PROVENANCE_LEVELS
    return DatasetVerification(local_ok, str(target.resolve()), hashes_ok, lock_integrity_ok, manifest_entries,
        reference_records, present, tuple(missing), provenance_level, computed_all_pinned,
        observed_unpinned_tuple, scientific, tuple(errors))
