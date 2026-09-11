from __future__ import annotations

from pathlib import Path
import tomllib

ROOT = Path(__file__).resolve().parents[1]


def test_integrated_optional_extras_are_declared():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]
    extras = project["optional-dependencies"]
    assert extras["conic"] == ["casadi==3.7.2"]
    assert extras["nlp"] == ["casadi==3.7.2"]
    assert extras["minlp"] == ["casadi==3.7.2"]
    assert extras["cp"] == ["ortools==9.15.6755"]


def test_release_qualification_requires_track_p_integrations_on_python_and_os_matrix():
    text = (ROOT / ".github/workflows/release-qualification.yml").read_text(encoding="utf-8")
    matrix = "extra: [highs, osqp, scip, nlopt, casadi, conic, nlp, minlp, cp, clarabel]"
    assert text.count(matrix) == 2
    assert 'python: ["3.12", "3.13", "3.14"]' in text
    assert 'os: [ubuntu-latest, macos-latest, windows-latest]' in text
    for extra in ("conic", "nlp", "minlp", "cp"):
        assert f'"{extra}"' in (ROOT / "tools/release_cell_runner.py").read_text(encoding="utf-8")


def test_release_smoke_has_semantic_checks_for_each_track_p_extra():
    text = (ROOT / "tools/release_smoke.py").read_text(encoding="utf-8")
    for extra in ("conic", "nlp", "minlp", "cp"):
        assert f'ns.require_extra == "{extra}"' in text


def test_current_public_docs_are_version_aligned():
    version = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]["version"]
    assert version == "0.3"
    for rel in (
        "README.md",
        "KNOWN-LIMITATIONS.md",
        "docs/api/EXTENDED-MODELING-API.md",
        "docs/release/SOLVERPILOT-PUBLIC-DOCS-0.3.md",
        "docs/release/README-CHECKLIST-0.3.md",
    ):
        assert version in (ROOT / rel).read_text(encoding="utf-8"), rel
