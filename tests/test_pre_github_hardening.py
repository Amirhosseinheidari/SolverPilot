from __future__ import annotations

import inspect
import json
import re
from pathlib import Path
import tomllib

import yaml
from packaging.version import Version

import solverpilot
from solverpilot import builtin_backend_candidates, default_registry

ROOT = Path(__file__).resolve().parents[1]
CI = ROOT / ".github/workflows/ci.yml"
QUAL = ROOT / ".github/workflows/release-qualification.yml"
PUBLISH = ROOT / ".github/workflows/publish.yml"
AUDIT = ROOT / "PRE-PUBLIC-RELEASE-AUDIT-0.1.0rc2.json"


def _sig(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def test_current_version_and_build_backend_are_exactly_frozen():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["project"]["version"] == solverpilot.__version__ == "0.1"
    assert Version(solverpilot.__version__).pre is None
    assert pyproject["build-system"]["requires"] == ["setuptools==84.0.0", "wheel==0.48.0"]


def test_current_public_api_snapshot_matches_runtime_without_surface_drift():
    payload = json.loads((ROOT / "PUBLIC-API-V0_1_0.json").read_text(encoding="utf-8"))
    assert payload["package_version"] == solverpilot.__version__
    assert [row["name"] for row in payload["symbols"]] == solverpilot.__all__
    assert len(solverpilot.__all__) == 78
    for row in payload["symbols"]:
        obj = getattr(solverpilot, row["name"])
        assert row["module"] == getattr(obj, "__module__", None)
        assert row["qualname"] == getattr(obj, "__qualname__", None)
        assert row["signature"] == _sig(obj)


def test_current_backend_contract_matches_runtime_and_keeps_learning_off():
    payload = json.loads((ROOT / "BACKEND-CONTRACT-V0_1_0.json").read_text(encoding="utf-8"))
    expected = {name for names in payload["stable_backend_ids"].values() for name in names}
    assert expected == {backend.manifest.name for backend in builtin_backend_candidates()}
    assert set(payload["stable_backend_ids"]["verification_only"]).isdisjoint(default_registry().names())
    assert payload["production_baseline_policy"]["learned_lp_performance_routing"] is False


def test_only_ci_qualification_and_manual_publication_are_active_workflows():
    active = sorted(path.name for path in (ROOT / ".github/workflows").glob("*.yml"))
    assert active == ["ci.yml", "publish.yml", "release-qualification.yml"]
    assert PUBLISH.is_file()
    frozen = ROOT / "docs/history/frozen-workflows"
    assert all((frozen / name).is_file() for name in ["m31-compatibility.yml", "m32-external-compatibility.yml", "m33-external-ci.yml"])


def test_normal_ci_is_lightweight_and_release_qualification_is_manual_only():
    ci_text = CI.read_text(encoding="utf-8")
    assert "push:" in ci_text and "pull_request:" in ci_text
    ci_jobs = yaml.safe_load(ci_text)["jobs"]
    assert ci_jobs["tests"]["strategy"]["matrix"]["python"] == ["3.12", "3.13", "3.14"]

    pre_permissions = QUAL.read_text(encoding="utf-8").split("permissions:", 1)[0]
    assert "workflow_dispatch:" in pre_permissions
    assert "push:" not in pre_permissions
    assert "pull_request:" not in pre_permissions


def test_every_active_or_publish_action_uses_full_commit_sha():
    for path in (CI, QUAL, PUBLISH):
        uses = re.findall(r"uses:\s*([^\s#]+)", path.read_text(encoding="utf-8"))
        assert uses
        for use in uses:
            assert re.fullmatch(r"[0-9a-f]{40}", use.rsplit("@", 1)[1]), use


def test_release_qualification_exact_pins_tooling_and_limits_write_permissions():
    text = QUAL.read_text(encoding="utf-8")
    assert '"build==1.6.0"' in text
    assert '"twine==7.0.0"' in text
    assert '"packaging==26.1"' in text
    assert "id-token: write" in text and "attestations: write" in text
    # Write permissions belong only to the dedicated optional attestation job.
    build_block = text.split("  build-dist:", 1)[1].split("  attest-dist:", 1)[0]
    assert "id-token: write" not in build_block
    assert "attestations: write" not in build_block


def test_publish_workflow_is_cross_run_commit_bound_attested_and_manual():
    text = PUBLISH.read_text(encoding="utf-8")
    for phrase in [
        "qualified_run_id", "qualified_commit_sha", "github-token:", "run-id:",
        "actions: read", "gh api", "release-qualification.yml", "gh attestation verify",
        "pypa/gh-action-pypi-publish@",
    ]:
        assert phrase in text
    assert "PYPI_API_TOKEN" not in text and "password:" not in text
    assert PUBLISH.suffix == ".yml"
    assert "workflow_dispatch:" in text
    assert "push:" not in text and "pull_request:" not in text
    assert "default: testpypi" in text
    assert "environment:" in text
    assert "packages-dir: publish-dist/" in text


def test_dependabot_covers_actions_and_python_dependency_metadata():
    rows = yaml.safe_load((ROOT / ".github/dependabot.yml").read_text(encoding="utf-8"))["updates"]
    ecosystems = {row["package-ecosystem"] for row in rows}
    assert {"github-actions", "pip"}.issubset(ecosystems)


def test_repository_does_not_embed_release_binary_artifacts():
    binary = []
    for pattern in ("*.whl", "*.tar.gz", "*.zip"):
        binary.extend(
            path for path in ROOT.rglob(pattern)
            if path.relative_to(ROOT).parts[0] not in {
                "dist", "dist-a", "dist-b", "dist-download", "build", ".venv"
            }
        )
    assert binary == []


def test_security_contribution_and_limitations_documents_are_present():
    for name in ["SECURITY.md", "CONTRIBUTING.md", "KNOWN-LIMITATIONS.md", "RELEASING.md"]:
        assert (ROOT / name).is_file()
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8")
    assert "Private Vulnerability Reporting" in security
    assert "Do **not** open a public issue" in security


def test_readme_has_one_current_release_state_not_stacked_milestones():
    text = (ROOT / "README.md").read_text(encoding="utf-8")
    head = "\n".join(text.splitlines()[:35])
    assert "0.1.0rc2" in head
    assert "M31 technical RC" not in head
    assert "M32 technical RC" not in head
    assert "M33 external-CI" not in head


def test_publication_fields_remain_uninvented_and_audit_fails_closed():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["maintainers"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["urls"] == {
        "Repository": "https://github.com/Amirhosseinheidari/SolverPilot",
        "Issues": "https://github.com/Amirhosseinheidari/SolverPilot/issues",
    }
    audit = json.loads(AUDIT.read_text(encoding="utf-8"))
    assert audit["public_github_open_source_authorized"] is False
    assert audit["pypi_public_release_authorized"] is False
    assert audit["public_1_0_authorized"] is False



def test_rc8_has_single_canonical_python_namespace_and_no_legacy_shim():
    assert not (ROOT / "src/optimind").exists()
    assert (ROOT / "src/solverpilot").is_dir()
    assert "optimind" not in {p.name for p in (ROOT / "src").iterdir() if p.is_dir()}


def test_release_tools_derive_current_version_and_rc_contracts_dynamically():
    manifest = (ROOT / "tools/release_dist_manifest.py").read_text(encoding="utf-8")
    smoke = (ROOT / "tools/release_smoke.py").read_text(encoding="utf-8")
    workflow = QUAL.read_text(encoding="utf-8")
    assert 'PROJECT = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]' in manifest
    assert 'EXPECTED_VERSION = PROJECT["version"]' in manifest
    assert 'CURRENT_RELEASE_LABEL = _release_label(EXPECTED_VERSION)' in manifest
    assert 'f"PUBLIC-API-{CURRENT_RELEASE_LABEL}.json"' in manifest
    assert 'f"BACKEND-CONTRACT-{CURRENT_RELEASE_LABEL}.json"' in manifest
    assert 'EXPECTED_VERSION = PROJECT["version"]' in smoke
    assert 'find_spec("optimind")' in smoke
    assert not re.search(r'EXPECTED_VERSION\s*=\s*"0\.0\.36rc\d+"', manifest + smoke)
    assert "build exact release artifacts" in workflow
    assert not re.search(r"build exact rc\d+ artifacts", workflow)

def test_apache_2_license_is_explicit_and_packaged():
    import tomllib
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    license_text = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert "Apache License" in license_text
    assert "Version 2.0, January 2004" in license_text
    manifest = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    assert "include LICENSE" in manifest
