from __future__ import annotations

import hashlib
import json
import os
from dataclasses import asdict
from pathlib import Path, PureWindowsPath
import shutil
import tarfile
import tempfile
import time
import urllib.request
from urllib.parse import urlparse
import zipfile

from .model import AcquisitionFile, AcquisitionRecord, DatasetSpec



def dataset_spec_fingerprint(spec: DatasetSpec) -> str:
    payload = asdict(spec)
    payload["instance_suffixes"] = list(spec.instance_suffixes)
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def sha256_file(path: str | Path, *, chunk_size: int = 1024 * 1024) -> str:
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        while True:
            chunk = f.read(chunk_size)
            if not chunk:
                break
            h.update(chunk)
    return h.hexdigest()


def _download(url: str, dest: Path, *, timeout_s: float = 60.0, retries: int = 3) -> None:
    if retries < 1:
        raise ValueError("retries must be >= 1")
    dest.parent.mkdir(parents=True, exist_ok=True)
    tmp = dest.with_name(dest.name + ".part")
    last_error: Exception | None = None
    for attempt in range(retries):
        offset = tmp.stat().st_size if tmp.exists() else 0
        headers = {"User-Agent": "solverpilot-benchmark/0.1"}
        if offset:
            headers["Range"] = f"bytes={offset}-"
        req = urllib.request.Request(url, headers=headers)
        try:
            with urllib.request.urlopen(req, timeout=timeout_s) as response:
                status = getattr(response, "status", None)
                append = bool(offset and status == 206)
                mode = "ab" if append else "wb"
                with tmp.open(mode) as out:
                    shutil.copyfileobj(response, out, length=1024 * 1024)
                    out.flush()
                    os.fsync(out.fileno())
            os.replace(tmp, dest)
            return
        except Exception as exc:
            last_error = exc
            if attempt + 1 < retries:
                time.sleep(min(4.0, 0.5 * (2**attempt)))
    if last_error is None:
        raise RuntimeError("download failed without a captured exception")
    raise last_error


def _safe_member_path(root: Path, member_name: str) -> Path:
    candidate = (root / member_name).resolve()
    root_resolved = root.resolve()
    if candidate != root_resolved and root_resolved not in candidate.parents:
        raise ValueError(f"archive member escapes target directory: {member_name!r}")
    return candidate


def _extract_archive(archive: Path, target: Path) -> int:
    """Extract an archive into a fresh directory and replace *target* only on success.

    Extracting directly into an existing directory can leave stale files behind when
    an upstream snapshot removes entries.  That is unacceptable for a benchmark
    corpus because a later run could silently mix two dataset snapshots.
    """

    target.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{target.name}.extract-", dir=target.parent))
    try:
        count = 0
        if zipfile.is_zipfile(archive):
            with zipfile.ZipFile(archive) as zf:
                for info in zf.infolist():
                    _safe_member_path(staging, info.filename)
                zf.extractall(staging)
                count = sum(not x.is_dir() for x in zf.infolist())
        elif tarfile.is_tarfile(archive):
            with tarfile.open(archive) as tf:
                members = tf.getmembers()
                for member in members:
                    _safe_member_path(staging, member.name)
                    if member.issym() or member.islnk():
                        raise ValueError("archive symlinks/hardlinks are not accepted")
                tf.extractall(staging, filter="data")
                count = sum(m.isfile() for m in members)
        else:
            raise ValueError(f"unsupported archive format: {archive}")

        if target.exists():
            shutil.rmtree(target)
        os.replace(staging, target)
        return count
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise


def _fetch_role(
    *, role: str, url: str | None, filename: str | None, expected_sha256: str | None, target: Path,
    timeout_s: float, retries: int, force: bool,
) -> AcquisitionFile | None:
    if url is None:
        return None
    name = filename or Path(urlparse(url).path).name
    if not name:
        raise ValueError(f"cannot derive filename from URL for {role}")
    if (name in {".", ".."} or "/" in name or "\\" in name or ":" in name
            or PureWindowsPath(name).is_absolute()):
        raise ValueError(f"download filename must be a single local filename: {name!r}")
    dest = target / name
    _safe_member_path(target, name)
    _safe_member_path(target, name + ".part")
    if force or not dest.exists():
        _download(url, dest, timeout_s=timeout_s, retries=retries)
    digest = sha256_file(dest)
    verified = expected_sha256 is not None and digest.lower() == expected_sha256.lower()
    if expected_sha256 is not None and not verified:
        raise ValueError(f"SHA-256 mismatch for {role}: expected {expected_sha256}, got {digest}")
    return AcquisitionFile(
        role=role,
        url=url,
        path=str(dest.resolve()),
        size_bytes=dest.stat().st_size,
        sha256=digest,
        expected_sha256=expected_sha256,
        verified=verified,
    )


def acquire_dataset(
    spec: DatasetSpec,
    target_dir: str | Path,
    *,
    extract: bool = True,
    timeout_s: float = 60.0,
    retries: int = 3,
    force: bool = False,
) -> AcquisitionRecord:
    target = Path(target_dir)
    target.mkdir(parents=True, exist_ok=True)
    files = []
    for role, url, filename, expected in (
        ("archive", spec.archive_url, spec.archive_filename, spec.archive_sha256),
        ("manifest", spec.manifest_url, spec.manifest_filename, spec.manifest_sha256),
        ("reference", spec.reference_url, spec.reference_filename, spec.reference_sha256),
    ):
        item = _fetch_role(
            role=role, url=url, filename=filename, expected_sha256=expected,
            target=target, timeout_s=timeout_s, retries=retries, force=force,
        )
        if item is not None:
            files.append(item)
    extracted_count = 0
    if extract:
        archive_file = next((f for f in files if f.role == "archive"), None)
        if archive_file is not None:
            extracted_count = _extract_archive(Path(archive_file.path), target / "instances")
    remote_roles = tuple(f.role for f in files)
    unpinned_roles = tuple(sorted(f.role for f in files if f.expected_sha256 is None))
    record = AcquisitionRecord(
        dataset=spec.name,
        spec_kind=spec.kind,
        target_dir=str(target.resolve()),
        files=tuple(files),
        extracted=bool(extract and any(f.role == "archive" for f in files)),
        extracted_files=extracted_count,
        metadata=dict(spec.metadata),
        schema_version="2.0",
        spec_fingerprint=dataset_spec_fingerprint(spec),
        provenance_level=spec.provenance_level,
        all_remote_inputs_pinned=bool(remote_roles) and not unpinned_roles,
        unpinned_roles=unpinned_roles,
    )
    lock = target / "acquisition-lock.json"
    lock.write_text(json.dumps(asdict(record), indent=2, sort_keys=True), encoding="utf-8")
    lock_digest = sha256_file(lock)
    (target / "acquisition-lock.sha256").write_text(
        f"{lock_digest}  {lock.name}\n", encoding="utf-8"
    )
    return record
