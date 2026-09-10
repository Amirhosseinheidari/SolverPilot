from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

CURRENT_BENCHMARK_ROW_SCHEMA = "1.5"


def canonical_sha256(value: Any) -> str:
    raw = json.dumps(value, sort_keys=True, separators=(",", ":"), ensure_ascii=False, default=str).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def experiment_identity(*, protocol_id: str, environment_id: str, backend_set_sha256: str, thread_policy: Mapping[str, Any]) -> str:
    return canonical_sha256({
        "protocol_id": protocol_id,
        "environment_id": environment_id,
        "backend_set_sha256": backend_set_sha256,
        "thread_policy": dict(thread_policy),
    })


def row_sha256(row: Mapping[str, Any]) -> str:
    return canonical_sha256({str(k): v for k, v in row.items() if k != "row_sha256"})


def attach_row_integrity(row: Mapping[str, Any]) -> dict[str, Any]:
    payload = dict(row)
    payload.pop("row_sha256", None)
    payload["row_sha256"] = row_sha256(payload)
    return payload


def verify_row_integrity(row: Mapping[str, Any], *, context: str = "benchmark row") -> None:
    schema = str(row.get("schema_version") or "")
    recorded = row.get("row_sha256")
    if recorded is None:
        if schema == CURRENT_BENCHMARK_ROW_SCHEMA:
            raise ValueError(f"{context}: schema {schema} row is missing row_sha256")
        return
    text = str(recorded)
    if len(text) != 64 or any(ch not in "0123456789abcdefABCDEF" for ch in text):
        raise ValueError(f"{context}: invalid row_sha256")
    if row_sha256(row).lower() != text.lower():
        raise ValueError(f"{context}: row_sha256 mismatch")
