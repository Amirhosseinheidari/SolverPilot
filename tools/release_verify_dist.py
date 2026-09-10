from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dist", type=Path)
    ns = ap.parse_args()
    dist = ns.dist.resolve()
    manifest = json.loads((dist / "RELEASE-DIST-MANIFEST.json").read_text())
    problems: list[str] = []
    for key in ("wheel", "sdist"):
        row = manifest[key]
        path = dist / row["filename"]
        if not path.is_file():
            problems.append(f"missing:{path.name}")
        elif sha256_file(path) != row["sha256"]:
            problems.append(f"sha256:{path.name}")
    print(json.dumps({"schema": "solverpilot.release.dist_verify.v1", "problems": problems, "passed": not problems}, sort_keys=True))
    return 0 if not problems else 2


if __name__ == "__main__":
    raise SystemExit(main())
