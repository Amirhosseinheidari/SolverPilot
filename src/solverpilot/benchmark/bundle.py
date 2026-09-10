from __future__ import annotations

import hashlib
import json
from pathlib import Path
import zipfile


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def create_result_bundle(
    output_zip: str | Path,
    *,
    files: list[str | Path],
    metadata: dict | None = None,
) -> dict:
    output = Path(output_zip)
    output.parent.mkdir(parents=True, exist_ok=True)
    entries = []
    seen_names = set()
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as zf:
        for raw in files:
            p = Path(raw)
            if not p.is_file():
                raise FileNotFoundError(p)
            arcname = p.name
            if arcname in seen_names:
                raise ValueError(f"duplicate bundle filename: {arcname}")
            seen_names.add(arcname)
            data = p.read_bytes()
            zf.writestr(arcname, data)
            entries.append({"name": arcname, "size_bytes": len(data), "sha256": sha256_bytes(data)})
        manifest = {"schema_version": "1.0", "metadata": metadata or {}, "entries": entries}
        zf.writestr("bundle-manifest.json", json.dumps(manifest, indent=2, sort_keys=True).encode())
    return {"path": str(output), "sha256": hashlib.sha256(output.read_bytes()).hexdigest(), "entries": entries}
