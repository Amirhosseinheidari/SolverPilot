from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path

EXPECTED_VERSION = "0.0.36rc3"
EXPECTED_NAME = "solverpilot"
EXPECTED_REQUIRES_PYTHON = ">=3.12"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dist", type=Path)
    ap.add_argument("--write", action="store_true")
    ns = ap.parse_args()
    dist = ns.dist.resolve()
    wheels = sorted(dist.glob("*.whl"))
    sdists = sorted(dist.glob("*.tar.gz"))
    if len(wheels) != 1 or len(sdists) != 1:
        raise SystemExit(f"expected exactly one wheel and one sdist, got {wheels=} {sdists=}")
    wheel, sdist = wheels[0], sdists[0]

    with zipfile.ZipFile(wheel) as z:
        wheel_bad = z.testzip()
        names = z.namelist()
        metadata_name = next(n for n in names if n.endswith(".dist-info/METADATA"))
        metadata = email.parser.Parser().parsestr(z.read(metadata_name).decode("utf-8"))
        pollution = [
            n for n in names
            if "__pycache__" in n or n.endswith((".pyc", ".pyo")) or n.startswith("tests/")
            or "/tests/" in n or ".egg-info/" in n or "/build/" in n
        ]
    with tarfile.open(sdist, "r:gz") as t:
        sdist_names = t.getnames()
        required = [
            "PUBLIC-API-M33.json",
            "BACKEND-CONTRACT-M33.json",
            "docs/release/M33-EXTERNAL-CI-EXECUTION-PROTOCOL.md",
            "docs/release/M33-PUBLICATION-SECURITY-AUDIT.md",
            "RELEASING.md",
        ]
        sdist_required = {item: any(n.endswith("/" + item) for n in sdist_names) for item in required}

    payload = {
        "schema": "solverpilot.m33.dist_manifest.v1",
        "name": metadata.get("Name"),
        "version": metadata.get("Version"),
        "requires_python": metadata.get("Requires-Python"),
        "wheel": {"filename": wheel.name, "size_bytes": wheel.stat().st_size, "sha256": sha256_file(wheel), "members": len(names), "testzip": wheel_bad},
        "sdist": {"filename": sdist.name, "size_bytes": sdist.stat().st_size, "sha256": sha256_file(sdist), "members": len(sdist_names)},
        "wheel_pollution": pollution,
        "sdist_required": sdist_required,
        "passed": metadata.get("Name") == EXPECTED_NAME
        and metadata.get("Version") == EXPECTED_VERSION
        and metadata.get("Requires-Python") == EXPECTED_REQUIRES_PYTHON
        and wheel_bad is None
        and not pollution
        and all(sdist_required.values()),
    }
    if ns.write:
        (dist / "M33-DIST-MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (dist / "SHA256SUMS-M33-DIST.txt").write_text(
            f"{payload['wheel']['sha256']}  {wheel.name}\n{payload['sdist']['sha256']}  {sdist.name}\n"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
