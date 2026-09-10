from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def load_lock(path: Path) -> dict:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("schema_version") != 1:
        raise ValueError("unsupported native-wheel-lock schema")
    packages = payload.get("packages")
    if not isinstance(packages, dict) or not packages:
        raise ValueError("native-wheel-lock must contain packages")
    for name, item in packages.items():
        for key in ("version", "filename", "url", "sha256", "size_bytes", "source"):
            if key not in item:
                raise ValueError(f"missing {key!r} for {name!r}")
        expected_hash = str(item["sha256"])
        if len(expected_hash) != 64 or any(c not in "0123456789abcdef" for c in expected_hash):
            raise ValueError(f"invalid sha256 for {name!r}")
    return payload


def verify_wheel_dir(lock: dict, wheel_dir: Path) -> dict[str, dict[str, object]]:
    results: dict[str, dict[str, object]] = {}
    for name, item in lock["packages"].items():
        path = wheel_dir / item["filename"]
        if not path.exists():
            results[name] = {"ok": False, "reason": "missing", "path": str(path)}
            continue
        actual_hash = sha256_file(path)
        actual_size = path.stat().st_size
        hash_ok = actual_hash == item["sha256"]
        expected_size = item.get("size_bytes")
        # PyPI UI rounds some displayed sizes. When exact bytes are unavailable, hash is authoritative.
        size_ok = True if expected_size is None else actual_size == int(expected_size)
        results[name] = {
            "ok": bool(hash_ok and size_ok),
            "hash_ok": hash_ok,
            "size_ok": size_ok,
            "actual_sha256": actual_hash,
            "expected_sha256": item["sha256"],
            "actual_size_bytes": actual_size,
            "expected_size_bytes": expected_size,
            "path": str(path),
        }
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description="Verify offline native solver wheels against the project lock file.")
    parser.add_argument("wheel_dir", type=Path)
    parser.add_argument(
        "--lock",
        type=Path,
        default=Path(__file__).resolve().parents[1] / "benchmarks" / "native-wheel-lock.json",
    )
    args = parser.parse_args()
    lock = load_lock(args.lock)
    results = verify_wheel_dir(lock, args.wheel_dir)
    print(json.dumps(results, indent=2, sort_keys=True))
    return 0 if all(item["ok"] for item in results.values()) else 2


if __name__ == "__main__":
    raise SystemExit(main())
