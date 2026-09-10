"""Safe deterministic local JSON ingestion."""

from __future__ import annotations

from pathlib import Path

from .errors import DataIOParseError, DataIOSourceError
from .hashing import sha256_bytes, sha256_json
from .mapping import apply_object_mapping
from .model import LoadPolicy, ObjectMapping
from .provenance import SourceFileProvenance, SourceProvenance
from .result import LoadedData
from .source import read_limited_bytes, validate_local_path
from .strict_json import loads_strict_json

_JSON_LOADER_VERSION = "1.0"


def load_json(
    source: str | Path,
    *,
    mapping: ObjectMapping | None = None,
    policy: LoadPolicy | None = None,
) -> LoadedData:
    """Load one local JSON file with strict parsing, mapping, and provenance.

    No problem type is inferred and no solver-specific object is materialized. The
    returned normalized payload is immutable; use ``mutable_payload()`` when a
    mutable detached copy is needed.
    """
    mapping = ObjectMapping.identity_mapping() if mapping is None else mapping
    policy = LoadPolicy() if policy is None else policy
    if not isinstance(mapping, ObjectMapping):
        from .errors import DataIOConfigurationError

        raise DataIOConfigurationError("mapping must be ObjectMapping.")
    if not isinstance(policy, LoadPolicy):
        from .errors import DataIOConfigurationError

        raise DataIOConfigurationError("policy must be LoadPolicy.")

    path = validate_local_path(
        source,
        expect_directory=False,
        reject_symlinks=policy.reject_symlinks,
    )
    raw_bytes = read_limited_bytes(path, max_bytes=policy.max_source_bytes)
    try:
        text = raw_bytes.decode(policy.encoding)
    except LookupError as exc:
        raise DataIOSourceError("Declared JSON encoding is not supported.") from exc
    except UnicodeDecodeError as exc:
        raise DataIOParseError("JSON source cannot be decoded with the declared encoding.") from exc

    parsed = loads_strict_json(
        text,
        context="JSON source",
        max_depth=policy.max_json_depth,
        strict_numbers=policy.strict_numbers,
    )
    normalized = apply_object_mapping(
        parsed,
        mapping,
        strict_numbers=policy.strict_numbers,
        max_json_depth=policy.max_json_depth,
    )
    source_hash = sha256_bytes(raw_bytes)
    provenance = SourceProvenance(
        loader="solverpilot.io.json",
        loader_version=_JSON_LOADER_VERSION,
        source_type="json_file",
        source_name=path.name,
        source_sha256=source_hash,
        source_size_bytes=len(raw_bytes),
        mapping_sha256=sha256_json(mapping.to_dict()),
        normalized_payload_sha256=sha256_json(normalized),
        policy_sha256=sha256_json(policy.to_dict()),
        record_count=len(parsed) if isinstance(parsed, list) else 1,
        encoding=policy.encoding,
        source_files=(
            SourceFileProvenance(name=path.name, sha256=source_hash, size_bytes=len(raw_bytes)),
        ),
        metadata={
            "mapping_version": mapping.version,
            "reject_symlinks": policy.reject_symlinks,
            "strict_numbers": policy.strict_numbers,
            "strict_json_policy": "solverpilot_strict_json_v1",
        },
    )
    return LoadedData(payload=normalized, provenance=provenance, metadata={"mapping_kind": "object"})
