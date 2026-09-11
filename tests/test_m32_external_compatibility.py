from __future__ import annotations

import hashlib
import inspect
import json
import re
from pathlib import Path
import tomllib

from packaging.version import Version
import yaml

import solverpilot
from solverpilot import builtin_backend_candidates, default_registry

ROOT = Path(__file__).resolve().parents[1]
PROTOCOL = ROOT / "docs/history/frozen-release-docs/M32-EXTERNAL-COMPATIBILITY-PROTOCOL.md"
WORKFLOW = ROOT / "docs/history/frozen-workflows/m32-external-compatibility.yml"
MATRIX = ROOT / "docs/history/frozen-m32/M32-COMPATIBILITY-MATRIX.json"
EXPECTED_PROTOCOL_SHA = "a0aa315e396ff983b6da472e75850dcf219d1c74d353a94efd1d5f09a18ffd29"


def _historical_name(name: str) -> str:
    return "SolverPilotError" if name == "OptiMindError" else name


def _historical_module(module: str | None) -> str | None:
    return module.replace("optimind.", "solverpilot.") if module else None


def _historical_qualname(name: str | None) -> str | None:
    return "SolverPilotError" if name == "OptiMindError" else name


def _sig(obj):
    try:
        from legacy_api_02 import legacy_signature
        return legacy_signature(obj)
    except (TypeError, ValueError):
        return None


def test_m32_protocol_was_frozen_and_hash_locked():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SHA
    payload = json.loads(MATRIX.read_text())
    assert payload["protocol_sha256"] == EXPECTED_PROTOCOL_SHA


def test_m32_version_is_coherent_pep440_rc2():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert json.loads(MATRIX.read_text())["package_version"] == "0.0.36rc2"
    parsed = Version("0.0.36rc2")
    assert parsed.is_prerelease and parsed.pre == ("rc", 2)


def test_m32_public_api_is_identical_to_m31_frozen_surface():
    m31 = json.loads((ROOT / "docs/history/frozen-m31/PUBLIC-API-M31.json").read_text())
    m32 = json.loads((ROOT / "docs/history/frozen-m32/PUBLIC-API-M32.json").read_text())
    assert m32["package_version"] == "0.0.36rc2"
    assert m31["symbols"] == m32["symbols"]
    assert [_historical_name(r["name"]) for r in m32["symbols"]] == solverpilot.__all__
    assert len(solverpilot.__all__) == 78
    for row in m32["symbols"]:
        obj = getattr(solverpilot, _historical_name(row["name"]))
        assert _historical_module(row["module"]) == getattr(obj, "__module__", None)
        assert _historical_qualname(row["qualname"]) == getattr(obj, "__qualname__", None)
        assert row["signature"] == _sig(obj)


def test_m32_backend_contract_matches_runtime_and_keeps_learning_off():
    payload = json.loads((ROOT / "docs/history/frozen-m32/BACKEND-CONTRACT-M32.json").read_text())
    ids = {name for names in payload["stable_backend_ids"].values() for name in names}
    assert ids == {b.manifest.name for b in builtin_backend_candidates()}
    assert set(payload["stable_backend_ids"]["verification_only"]).isdisjoint(default_registry().names())
    assert payload["production_baseline_policy"]["learned_lp_performance_routing"] is False


def test_m32_workflow_core_matrix_is_3_os_by_4_python():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    matrix = jobs["core-matrix"]["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert matrix["python"] == ["3.10", "3.11", "3.12", "3.13"]
    native_py = jobs["native-python-matrix"]["strategy"]["matrix"]
    assert native_py["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert set(native_py["extra"]) == {"highs", "osqp", "scip", "nlopt", "casadi"}


def test_m32_workflow_builds_once_and_tests_downloaded_exact_wheel():
    text = WORKFLOW.read_text()
    jobs = yaml.safe_load(text)["jobs"]
    assert "build-dist" in jobs
    assert "python -m build --outdir dist" in text
    assert "m32_dist_manifest.py dist --write" in text
    assert "m32_verify_dist.py dist" in text
    assert "m32_install_wheel.py --dist dist" in text
    assert "actions/download-artifact@" in text
    # Matrix jobs must not install the project source checkout directly.
    for job_name in ("core-matrix", "native-python-matrix", "native-os-matrix"):
        dumped = json.dumps(jobs[job_name])
        assert 'pip install ".[test' not in dumped
        assert 'pip install ".[native' not in dumped
        assert "m32_install_wheel.py" in dumped


def test_every_external_github_action_is_pinned_to_full_sha():
    uses = re.findall(r"uses:\s*([^\s#]+)", WORKFLOW.read_text())
    assert uses
    for use in uses:
        assert "@" in use
        ref = use.rsplit("@", 1)[1]
        assert re.fullmatch(r"[0-9a-f]{40}", ref), use


def test_m32_workflow_records_resolver_pip_check_and_junit():
    text = WORKFLOW.read_text()
    assert "m32_resolver_report.py" in text
    assert "m32-pytest.xml" in text
    assert "m32-smoke.json" in text
    assert "technical-compatibility-gate" in text


def test_m32_workflow_prepares_attestation_but_has_no_publish_action():
    text = WORKFLOW.read_text()
    assert "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d" in text
    assert "gh-action-pypi-publish" not in text
    assert "PYPI_API_TOKEN" not in text


def test_owner_publication_fields_remain_uninvented():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["maintainers"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["urls"] == {
        "Repository": "https://github.com/Amirhosseinheidari/SolverPilot",
        "Issues": "https://github.com/Amirhosseinheidari/SolverPilot/issues",
    }


def test_m32_release_extra_contains_validation_tools():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    release = project["optional-dependencies"]["release"]
    assert any(x.startswith("build>=") for x in release)
    assert any(x.startswith("twine>=") for x in release)
    assert any(x.startswith("packaging>=") for x in release)


def test_m32_public_gate_is_fail_closed_until_execution_and_owner_inputs():
    payload = json.loads(MATRIX.read_text())
    assert payload["public_rc_authorized"] is False
    assert payload["public_1_0_authorized"] is False
    assert payload["publish_workflow_enabled"] is False
    assert len(payload["core_matrix"]) == 12
    assert all(row["status"] in {"pending_external_execution", "executed_pass"} for row in payload["core_matrix"])
