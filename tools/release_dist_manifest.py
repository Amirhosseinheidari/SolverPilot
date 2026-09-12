from __future__ import annotations

import argparse
import email.parser
import hashlib
import json
import re
import tarfile
import tomllib
import zipfile

from packaging.specifiers import SpecifierSet
from packaging.version import Version
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
EXPECTED_VERSION = PROJECT["version"]
EXPECTED_NAME = PROJECT["name"]
EXPECTED_REQUIRES_PYTHON = PROJECT["requires-python"]
EXPECTED_LICENSE = PROJECT.get("license")


def _release_label(version: str) -> str:
    parsed = Version(version)
    if parsed.epoch or parsed.post is not None or parsed.dev is not None or parsed.local is not None:
        raise RuntimeError(f"unsupported release version: {version!r}")
    if parsed.pre is None:
        return f"V{parsed.major}_{parsed.minor}_{parsed.micro}"
    if parsed.pre[0] != "rc":
        raise RuntimeError(f"release tooling supports final and rc versions, got {version!r}")
    return f"V{parsed.major}_{parsed.minor}_{parsed.micro}RC{parsed.pre[1]}"


CURRENT_RELEASE_LABEL = _release_label(EXPECTED_VERSION)
CURRENT_RELEASE_DOC_VERSION = EXPECTED_VERSION.upper()


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
        canonical_package_present = any(n.startswith("solverpilot/") for n in names)
        typed_marker_present = 'solverpilot/py.typed' in names
        legacy_package_members = [n for n in names if n.startswith("optimind/")]
        license_entries = [n for n in names if "/licenses/LICENSE" in n or n.endswith("/LICENSE")]

    with tarfile.open(sdist, "r:gz") as t:
        sdist_names = t.getnames()
        required = [
            f"PUBLIC-API-{CURRENT_RELEASE_LABEL}.json",
            f"BACKEND-CONTRACT-{CURRENT_RELEASE_LABEL}.json",
            "README.md",
            "LICENSE",
            "SECURITY.md",
            "CONTRIBUTING.md",
            "KNOWN-LIMITATIONS.md",
            "RELEASING.md",
            "TRACK-P-MERGE-PROVENANCE.json",
            # Frozen historical evidence; the current artifacts are qualified separately.
            "PRE-PUBLIC-RELEASE-AUDIT-0.1.0rc2.json",
            "PRE-PUBLIC-RELEASE-VERIFICATION-0.1.0rc2.md",
            "docs/release/PRE-GITHUB-HARDENING.md",
            f"docs/release/SOLVERPILOT-PUBLIC-DOCS-{CURRENT_RELEASE_DOC_VERSION}.md",
            f"docs/release/README-CHECKLIST-{CURRENT_RELEASE_DOC_VERSION}.md",
            "docs/api/PUBLIC-API-v1.md",
            "docs/api/EXTENDED-MODELING-API.md",
            "examples/README.md",
            "examples/01_lp_basic.py",
            "examples/02_milp_binary.py",
            "examples/03_qp_convex.py",
            "examples/04_choose_backend.py",
            "examples/05_infeasible.py",
            "examples/06_validation_diagnostics.py",
            "examples/07_reoptimization_session.py",
            "examples/08_semantic_model.py",
            "examples/09_conic_model.py",
            "examples/10_nlp_optional.py",
            "examples/11_minlp_optional.py",
            "examples/12_cp_reference.py",
            "examples/13_cp_sat_optional.py",
            "examples/14_persistent_session_optional.py",
            "examples/15_named_model_and_quality.py",
            "examples/16_direct_conic.py",
            "examples/17_batch_scenarios.py",
            "examples/18_sensitivity_and_scenarios.py",
            "examples/19_convex_atoms_optional.py",
            "examples/20_streaming_reuse.py",
            "examples/21_checked_run_replay.py",
            "examples/22_global_optimization_optional.py",
            "examples/23_generalized_power_optional.py",
            "examples/24_pdlp_and_certificates_optional.py",
        ]
        sdist_required = {item: any(n.endswith("/" + item) for n in sdist_names) for item in required}
        legacy_sdist_members = [n for n in sdist_names if "/optimind/" in n or n.endswith("/optimind")]

    payload = {
        "schema": "solverpilot.release.dist_manifest.v2",
        "project_source": {
            "name": EXPECTED_NAME,
            "version": EXPECTED_VERSION,
            "requires_python": EXPECTED_REQUIRES_PYTHON,
            "license": EXPECTED_LICENSE,
            "release_label": CURRENT_RELEASE_LABEL,
        },
        "name": metadata.get("Name"),
        "version": metadata.get("Version"),
        "requires_python": metadata.get("Requires-Python"),
        "license_expression": metadata.get("License-Expression"),
        "wheel": {
            "filename": wheel.name,
            "size_bytes": wheel.stat().st_size,
            "sha256": sha256_file(wheel),
            "members": len(names),
            "testzip": wheel_bad,
        },
        "sdist": {
            "filename": sdist.name,
            "size_bytes": sdist.stat().st_size,
            "sha256": sha256_file(sdist),
            "members": len(sdist_names),
        },
        "wheel_pollution": pollution,
        "canonical_package_present": canonical_package_present,
        "typed_marker_present": typed_marker_present,
        "legacy_package_members": legacy_package_members,
        "legacy_sdist_members": legacy_sdist_members,
        "license_entries": license_entries,
        "sdist_required": sdist_required,
    }
    payload["passed"] = bool(
        metadata.get("Name") == EXPECTED_NAME
        and metadata.get("Version") == EXPECTED_VERSION
        and SpecifierSet(metadata.get("Requires-Python") or "") == SpecifierSet(EXPECTED_REQUIRES_PYTHON)
        and metadata.get("License-Expression") == EXPECTED_LICENSE
        and wheel_bad is None
        and not pollution
        and canonical_package_present
        and typed_marker_present
        and not legacy_package_members
        and not legacy_sdist_members
        and bool(license_entries)
        and all(sdist_required.values())
    )
    if ns.write:
        (dist / "RELEASE-DIST-MANIFEST.json").write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
        (dist / "SHA256SUMS-RELEASE-DIST.txt").write_text(
            f"{payload['wheel']['sha256']}  {wheel.name}\n{payload['sdist']['sha256']}  {sdist.name}\n"
        )
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if payload["passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
