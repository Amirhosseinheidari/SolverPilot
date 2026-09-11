from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
import tomllib

import solverpilot
from solverpilot import builtin_backend_candidates, default_registry

ROOT = Path(__file__).resolve().parents[1]


def _sig(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def test_rc8_identity_and_version_are_canonical():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["name"] == "solverpilot"
    assert project["version"] == solverpilot.__version__ == "0.1"
    assert project["license"] == "Apache-2.0"
    assert project["authors"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["maintainers"] == [{"name": "Amirhossein Heidari Rashtabad"}]


def test_rc8_public_api_snapshot_exactly_matches_runtime():
    payload = json.loads((ROOT / "PUBLIC-API-V0_1_0.json").read_text())
    assert payload["package_version"] == solverpilot.__version__
    assert payload["schema"] == "solverpilot.public_api.v1"
    assert [r["name"] for r in payload["symbols"]] == solverpilot.__all__
    assert len(solverpilot.__all__) == 78
    for row in payload["symbols"]:
        obj = getattr(solverpilot, row["name"])
        assert row["module"] == getattr(obj, "__module__", None)
        assert row["qualname"] == getattr(obj, "__qualname__", None)
        assert row["signature"] == _sig(obj)


def test_rc8_backend_contract_matches_runtime_and_learning_stays_off():
    payload = json.loads((ROOT / "BACKEND-CONTRACT-V0_1_0.json").read_text())
    expected = {name for group in payload["stable_backend_ids"].values() for name in group}
    assert expected == {b.manifest.name for b in builtin_backend_candidates()}
    assert set(payload["stable_backend_ids"]["verification_only"]).isdisjoint(default_registry().names())
    assert payload["production_baseline_policy"]["learned_lp_performance_routing"] is False


def test_rc8_distribution_source_contains_no_optimind_package_or_exception_alias():
    assert not (ROOT / "src/optimind").exists()
    assert "OptiMindError" not in (ROOT / "src/solverpilot/exceptions.py").read_text()


def test_rc8_release_tools_have_no_prerelease_number_hardcode():
    text = (ROOT / "tools/release_dist_manifest.py").read_text() + (ROOT / "tools/release_smoke.py").read_text()
    assert not re.search(r'EXPECTED_VERSION\\s*=\\s*"0\\.0\\.36rc\\d+"', text)
    assert 'EXPECTED_VERSION = PROJECT["version"]' in text
    assert 'find_spec("optimind")' in text


def test_rc8_current_manifest_requires_rc8_contracts_not_rc6():
    manifest = (ROOT / "MANIFEST.in").read_text()
    assert "include PUBLIC-API-V0_1_0.json" in manifest
    assert "include BACKEND-CONTRACT-V0_1_0.json" in manifest
    assert "PUBLIC-API-RC6.json" not in manifest
    assert "BACKEND-CONTRACT-RC6.json" not in manifest


def test_rc8_live_tools_only_contain_current_release_helpers():
    names = sorted(p.name for p in (ROOT / "tools").glob("*.py"))
    assert names == [
        "release_cell_runner.py",
        "release_dist_manifest.py",
        "release_repro_build_compare.py",
        "release_resolver_report.py",
        "release_runtime_requirements.py",
        "release_smoke.py",
        "release_verify_dist.py",
        "verify_native_wheels.py",
    ]
