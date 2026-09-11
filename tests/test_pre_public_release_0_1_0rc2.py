from __future__ import annotations

import json
from pathlib import Path
import tomllib

from packaging.version import Version
import solverpilot

ROOT = Path(__file__).resolve().parents[1]


def test_final_release_identity_is_distinct_from_historical_rc2():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["version"] == solverpilot.__version__ == "0.2"
    assert Version(project["version"]).pre is None
    assert project["version"] != "0.0.40rc2"


def test_current_release_snapshots_are_versioned_without_label_collision():
    api = json.loads((ROOT / "PUBLIC-API-V0_2_0.json").read_text(encoding="utf-8"))
    backend = json.loads((ROOT / "BACKEND-CONTRACT-V0_2_0.json").read_text(encoding="utf-8"))
    assert api["package_version"] == "0.2"
    assert backend["package_version"] == "0.2"
    assert len(api["symbols"]) == 78
    assert [row["name"] for row in api["symbols"]] == solverpilot.__all__


def test_historical_track_p_provenance_remains_historical():
    payload = json.loads((ROOT / "TRACK-P-MERGE-PROVENANCE.json").read_text(encoding="utf-8"))
    assert payload["integration"]["version"] == "0.0.40rc2"
    assert payload["source_bundle"]["track_p_version"] == "0.0.40"


def test_release_manifest_uses_full_semantic_version_label():
    text = (ROOT / "tools/release_dist_manifest.py").read_text(encoding="utf-8")
    assert 'return f"V{parsed.major}_{parsed.minor}_{parsed.micro}RC{parsed.pre[1]}"' in text
    assert 'CURRENT_RELEASE_LABEL = _release_label(EXPECTED_VERSION)' in text
    assert 'f"PUBLIC-API-{CURRENT_RELEASE_LABEL}.json"' in text
    assert 'f"BACKEND-CONTRACT-{CURRENT_RELEASE_LABEL}.json"' in text
    assert 'CURRENT_RELEASE_DOC_VERSION = EXPECTED_VERSION.upper()' in text


def test_manifest_contains_current_release_contracts_not_historical_current_contracts():
    text = (ROOT / "MANIFEST.in").read_text(encoding="utf-8")
    for required in (
        "PUBLIC-API-V0_1_0RC2.json",
        "BACKEND-CONTRACT-V0_1_0RC2.json",
        "PRE-PUBLIC-RELEASE-AUDIT-0.1.0rc2.json",
        "PRE-PUBLIC-RELEASE-VERIFICATION-0.1.0rc2.md",
        "docs/release/SOLVERPILOT-PUBLIC-DOCS-0.1.0RC2.md",
        "docs/release/README-CHECKLIST-0.1.0RC2.md",
    ):
        assert f"include {required}" in text


def test_pre_public_audit_remains_fail_closed():
    audit = json.loads((ROOT / "PRE-PUBLIC-RELEASE-AUDIT-0.1.0rc2.json").read_text(encoding="utf-8"))
    assert audit["version"] == "0.1.0rc2"
    assert audit["pypi_name_check"]["result"] == "404_not_found_at_check_time"
    assert audit["pypi_public_release_authorized"] is False
    assert audit["publication_blockers"]


def test_current_docs_describe_final_release_as_unchanged_promotion():
    readme = (ROOT / "README.md").read_text(encoding="utf-8")
    head = "\n".join(readme.splitlines()[:55])
    assert "0.1.0rc2" in head
    assert "0.2" in head
    assert "correctness/trust hardening" in head
    assert "package version intentionally remains `0.0.40rc2`" not in readme


def test_build_and_optional_runtime_pins_remain_exact_for_external_qualification():
    pyproject = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert pyproject["build-system"]["requires"] == ["setuptools==84.0.0", "wheel==0.48.0"]
    assert pyproject["project"]["optional-dependencies"]["cp"] == ["ortools==9.15.6755"]
    workflow = (ROOT / ".github/workflows/release-qualification.yml").read_text(encoding="utf-8")
    for pin in ('"build==1.6.0"', '"twine==7.0.0"', '"packaging==26.1"'):
        assert pin in workflow


def test_no_project_urls_are_invented_before_owner_approval():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    assert project["urls"] == {
        "Repository": "https://github.com/Amirhosseinheidari/SolverPilot",
        "Issues": "https://github.com/Amirhosseinheidari/SolverPilot/issues",
    }


def test_top_level_api_is_still_frozen():
    assert len(solverpilot.__all__) == 78


def test_release_label_is_collision_resistant_across_major_minor_patch():
    import importlib.util
    spec = importlib.util.spec_from_file_location("release_dist_manifest_test", ROOT / "tools/release_dist_manifest.py")
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    assert module._release_label("0.1") == "V0_1_0"
    assert module._release_label("0.1.0") == "V0_1_0"
    assert module._release_label("0.1") != module._release_label("0.1.0rc2")
    assert module._release_label("0.1.0rc2") == "V0_1_0RC2"
    assert module._release_label("0.0.40rc2") == "V0_0_40RC2"
    assert module._release_label("1.0.0rc1") == "V1_0_0RC1"
    assert len({module._release_label(v) for v in ("0.1.0rc2", "0.0.40rc2", "1.0.0rc1")}) == 3


def test_release_cell_runner_has_explicit_installed_wheel_example_gate():
    text = (ROOT / "tools/release_cell_runner.py").read_text(encoding="utf-8")
    assert "def clean_runtime_env" in text
    assert 'env.pop("PYTHONPATH", None)' in text
    assert 'env["PYTHONNOUSERSITE"] = "1"' in text
    assert "def run_installed_public_examples" in text
    assert 'glob("[0-9][0-9]_*.py")' in text
    assert "expected 17 public examples" in text
    assert "run_installed_public_examples(py" in text


def test_dist_manifest_preserves_historical_pre_public_evidence():
    text = (ROOT / "tools/release_dist_manifest.py").read_text(encoding="utf-8")
    assert '"PRE-PUBLIC-RELEASE-AUDIT-0.1.0rc2.json"' in text
    assert '"PRE-PUBLIC-RELEASE-VERIFICATION-0.1.0rc2.md"' in text
