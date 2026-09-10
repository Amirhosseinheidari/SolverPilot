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
PROTOCOL = ROOT / "docs/history/frozen-release-docs/M33-EXTERNAL-CI-EXECUTION-PROTOCOL.md"
WORKFLOW = ROOT / "docs/history/frozen-workflows/m33-external-ci.yml"
PUBLISH_TEMPLATE = ROOT / "docs/history/frozen-workflows/m33-publish.yml.disabled"
MATRIX = ROOT / "docs/history/frozen-m33/M33-COMPATIBILITY-MATRIX.json"
EXPECTED_PROTOCOL_SHA = "041dbc772f54a60d2dab451c4b34083e4d61c0e3bec6ee68c43cd54e0799921c"


def _historical_name(name: str) -> str:
    return "SolverPilotError" if name == "OptiMindError" else name


def _historical_module(module: str | None) -> str | None:
    return module.replace("optimind.", "solverpilot.") if module else None


def _historical_qualname(name: str | None) -> str | None:
    return "SolverPilotError" if name == "OptiMindError" else name


def _sig(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def test_m33_protocol_is_hash_frozen_before_implementation():
    assert hashlib.sha256(PROTOCOL.read_bytes()).hexdigest() == EXPECTED_PROTOCOL_SHA
    assert json.loads(MATRIX.read_text())["protocol_sha256"] == EXPECTED_PROTOCOL_SHA


def test_m33_version_and_modern_support_floor_are_coherent():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert json.loads(MATRIX.read_text())["package_version"] == "0.0.36rc3"
    assert Version("0.0.36rc3").pre == ("rc", 3)
    assert Version(project["version"]) > Version("0.0.36rc3")
    assert project["requires-python"] == ">=3.12,<3.15"
    assert "numpy>=2.2,<3" in project["dependencies"]
    assert "scipy>=1.15,<2" in project["dependencies"]
    assert all(f"Programming Language :: Python :: {v}" in project["classifiers"] for v in ("3.12", "3.13", "3.14"))


def test_m33_public_api_has_zero_drift_from_m32():
    m32 = json.loads((ROOT / "docs/history/frozen-m32/PUBLIC-API-M32.json").read_text())
    m33 = json.loads((ROOT / "docs/history/frozen-m33/PUBLIC-API-M33.json").read_text())
    assert m33["package_version"] == "0.0.36rc3"
    assert m32["symbols"] == m33["symbols"]
    assert [_historical_name(r["name"]) for r in m33["symbols"]] == solverpilot.__all__
    assert len(solverpilot.__all__) == 78
    for row in m33["symbols"]:
        obj = getattr(solverpilot, _historical_name(row["name"]))
        assert _historical_module(row["module"]) == getattr(obj, "__module__", None)
        assert _historical_qualname(row["qualname"]) == getattr(obj, "__qualname__", None)
        assert row["signature"] == _sig(obj)


def test_m33_backend_contract_has_zero_runtime_drift_and_learning_stays_off():
    payload = json.loads((ROOT / "docs/history/frozen-m33/BACKEND-CONTRACT-M33.json").read_text())
    ids = {name for names in payload["stable_backend_ids"].values() for name in names}
    assert ids == {b.manifest.name for b in builtin_backend_candidates()}
    assert set(payload["stable_backend_ids"]["verification_only"]).isdisjoint(default_registry().names())
    assert payload["production_baseline_policy"]["learned_lp_performance_routing"] is False


def test_m33_core_matrix_is_3_os_by_python_312_to_314_only():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    matrix = jobs["core-matrix"]["strategy"]["matrix"]
    assert matrix["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    assert matrix["python"] == ["3.12", "3.13", "3.14"]
    dumped = WORKFLOW.read_text()
    assert 'python: ["3.10"' not in dumped
    assert 'python: ["3.11"' not in dumped
    payload = json.loads(MATRIX.read_text())
    assert len(payload["core_matrix"]) == 9
    assert payload["core_pending_count"] == 9
    assert payload["core_executed_pass_count"] == 0


def test_m33_minimum_dependency_job_is_explicit_and_full_regression():
    jobs = yaml.safe_load(WORKFLOW.read_text())["jobs"]
    job = jobs["minimum-deps"]
    dumped = json.dumps(job)
    assert 'python-version": "3.12"' in dumped
    assert "m33_cell_runner.py --dist dist --prefix m33-min --minimum-deps" in dumped
    helper = (ROOT / "docs/history/frozen-tools/m33_cell_runner.py").read_text()
    assert 'MIN_NUMPY = "2.2.0"' in helper
    assert 'MIN_SCIPY = "1.15.0"' in helper
    assert "--skip-regression" not in dumped


def test_m33_cells_use_isolated_exact_wheel_and_runtime_security_audit():
    text = WORKFLOW.read_text()
    assert "m33_verify_dist.py dist" in text
    assert "m33_cell_runner.py --dist dist" in text
    runner = (ROOT / "docs/history/frozen-tools/m33_cell_runner.py").read_text()
    assert 'PIP_AUDIT_VERSION = "2.10.1"' in runner
    assert '"cyclonedx-json"' in runner
    assert '"-m", "venv"' in runner
    assert '"-m", "pip", "check"' not in runner  # resolver helper owns pip check and fails closed
    resolver = (ROOT / "docs/history/frozen-tools/m33_resolver_report.py").read_text()
    assert '"-m", "pip", "check"' in resolver
    assert "pip_audit" in runner


def test_m33_builds_twice_with_fixed_epoch_and_attests_then_verifies():
    text = WORKFLOW.read_text()
    assert "SOURCE_DATE_EPOCH=$(git log -1 --pretty=%ct)" in text
    assert "python -m build --outdir dist-a" in text
    assert "python -m build --outdir dist-b" in text
    assert "m33_repro_build_compare.py" in text
    assert "python -m twine check --strict dist/*" in text
    assert "actions/attest@508db95dd578ae2727ebd6217d5ba78e4fbda05d" in text
    assert "gh attestation verify" in text


def test_all_m33_external_actions_and_publish_template_actions_use_full_sha():
    for path in (WORKFLOW, PUBLISH_TEMPLATE):
        uses = re.findall(r"uses:\s*([^\s#]+)", path.read_text())
        assert uses
        for use in uses:
            assert re.fullmatch(r"[0-9a-f]{40}", use.rsplit("@", 1)[1]), use


def test_m33_dependabot_tracks_github_actions_for_reviewed_sha_updates():
    payload = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text())
    rows = payload["updates"]
    assert any(row["package-ecosystem"] == "github-actions" and row["schedule"]["interval"] == "monthly" for row in rows)


def test_m33_publish_template_is_disabled_oidc_only_and_no_long_lived_token():
    assert PUBLISH_TEMPLATE.suffix == ".disabled"
    text = PUBLISH_TEMPLATE.read_text()
    assert "id-token: write" in text
    assert "pypa/gh-action-pypi-publish@dc37677b2e1c63e2034f94d8a5b11f265b73ba33" in text
    assert "PYPI_API_TOKEN" not in text
    assert "password:" not in text
    assert "environment:" in text and "name: pypi" in text


def test_m33_release_process_documents_external_repository_security_state():
    text = (ROOT / "RELEASING.md").read_text()
    for phrase in ["branch protection", "required reviewer", "Trusted Publisher", "allowed-actions", "long-lived PyPI API token"]:
        assert phrase in text


def test_owner_metadata_stays_uninvented_and_public_gate_stays_false():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["maintainers"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert "urls" not in project
    payload = json.loads(MATRIX.read_text())
    assert payload["technical_external_gate"] is False
    assert payload["public_rc_authorized"] is False
    assert payload["public_1_0_authorized"] is False
    assert payload["publish_workflow_enabled"] is False
    assert payload["security_gate"]["ignore_list"] == []
    assert all(v.startswith("unverified") or v.startswith("unconfigured") for v in payload["external_repository_settings"].values())
