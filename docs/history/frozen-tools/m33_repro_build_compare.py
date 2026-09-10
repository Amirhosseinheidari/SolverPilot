from __future__ import annotations

import argparse
import hashlib
import json
import tarfile
import zipfile
from pathlib import Path


def sha_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha_file(path: Path) -> str:
    return sha_bytes(path.read_bytes())


def wheel_members(path: Path) -> dict[str, str]:
    with zipfile.ZipFile(path) as z:
        return {n: sha_bytes(z.read(n)) for n in z.namelist()}


def sdist_members(path: Path) -> dict[str, str]:
    out: dict[str, str] = {}
    with tarfile.open(path, "r:gz") as t:
        for member in t.getmembers():
            if member.isfile():
                f = t.extractfile(member)
                assert f is not None
                out[member.name] = sha_bytes(f.read())
    return out


def compare_maps(a: dict[str, str], b: dict[str, str]) -> dict[str, object]:
    names_a, names_b = set(a), set(b)
    changed = sorted(name for name in names_a & names_b if a[name] != b[name])
    return {
        "names_equal": names_a == names_b,
        "only_a": sorted(names_a - names_b),
        "only_b": sorted(names_b - names_a),
        "changed": changed,
        "content_equal": names_a == names_b and not changed,
    }


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("dist_a", type=Path)
    ap.add_argument("dist_b", type=Path)
    ap.add_argument("--output", type=Path, required=True)
    ns = ap.parse_args()
    wa = next(ns.dist_a.glob("*.whl")); wb = next(ns.dist_b.glob("*.whl"))
    sa = next(ns.dist_a.glob("*.tar.gz")); sb = next(ns.dist_b.glob("*.tar.gz"))
    wheel_cmp = compare_maps(wheel_members(wa), wheel_members(wb))
    sdist_cmp = compare_maps(sdist_members(sa), sdist_members(sb))
    payload = {
        "schema": "solverpilot.m33.repro_build.v1",
        "wheel": {"sha_a": sha_file(wa), "sha_b": sha_file(wb), "byte_identical": sha_file(wa) == sha_file(wb), **wheel_cmp},
        "sdist": {"sha_a": sha_file(sa), "sha_b": sha_file(sb), "byte_identical": sha_file(sa) == sha_file(sb), **sdist_cmp},
        "semantic_content_equal": bool(wheel_cmp["content_equal"] and sdist_cmp["content_equal"]),
    }
    ns.output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["semantic_content_equal"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
