from __future__ import annotations

import os
from pathlib import Path
import re
import subprocess
import sys
import tomllib

ROOT = Path(__file__).resolve().parents[1]
README = ROOT / "README.md"
EXAMPLES = ROOT / "examples"


def test_readme_covers_requested_public_user_path():
    text = README.read_text()
    required = [
        "SolverPilot is a **trust-aware optimization runtime and modeling layer for Python**",
        "Python **3.12, 3.13, or 3.14**",
        "python -m pip install solverpilot",
        "## Quick Start",
        "### Linear Programming (LP)",
        "### Mixed-Integer Linear Programming (MILP)",
        "### Continuous convex QP",
        "## Choosing a backend",
        "## Understanding `SolveResult`",
        "## Infeasible problems and diagnostics",
        "## Repeated solves and persistence",
        "## Optional solver dependencies",
        "## What is not supported or not claimed yet",
        "## Examples",
        "## API Reference",
        "[Stable top-level Public API](docs/api/PUBLIC-API-v1.md)",
        "[Extended Modeling API](docs/api/EXTENDED-MODELING-API.md)",
    ]
    for phrase in required:
        assert phrase in text, phrase


def test_readme_installation_does_not_claim_pypi_is_already_live():
    text = README.read_text()
    install = text.split("## Installation", 1)[1].split("## Quick Start", 1)[0]
    assert "After the first public release" in install
    assert "git clone <REPOSITORY_URL>" in install


def test_readme_optional_extras_match_pyproject():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    extras = set(project["optional-dependencies"])
    text = README.read_text()
    for extra in {"highs", "osqp", "scip", "nlopt", "casadi", "native", "open-source", "benchmark", "conic", "nlp", "minlp", "cp"}:
        assert extra in extras
        assert f"solverpilot[{extra}]" in text


def test_readme_relative_links_resolve():
    text = README.read_text()
    targets = re.findall(r"\[[^\]]+\]\(([^)]+)\)", text)
    missing: list[str] = []
    for target in targets:
        if target.startswith(("http://", "https://", "mailto:", "#")):
            continue
        rel = target.split("#", 1)[0]
        if rel and not (ROOT / rel).exists():
            missing.append(target)
    assert missing == []


def test_expected_public_examples_exist():
    names = sorted(p.name for p in EXAMPLES.glob("[0-9][0-9]_*.py"))
    assert names == [
        "01_lp_basic.py",
        "02_milp_binary.py",
        "03_qp_convex.py",
        "04_choose_backend.py",
        "05_infeasible.py",
        "06_validation_diagnostics.py",
        "07_reoptimization_session.py",
        "08_semantic_model.py",
        "09_conic_model.py",
        "10_nlp_optional.py",
        "11_minlp_optional.py",
        "12_cp_reference.py",
        "13_cp_sat_optional.py",
        "14_persistent_session_optional.py",
    ]
    assert (EXAMPLES / "README.md").is_file()


def test_all_public_examples_execute_against_current_source():
    env = os.environ.copy()
    src = str(ROOT / "src")
    env["PYTHONPATH"] = src + (os.pathsep + env["PYTHONPATH"] if env.get("PYTHONPATH") else "")
    for path in sorted(EXAMPLES.glob("[0-9][0-9]_*.py")):
        completed = subprocess.run(
            [sys.executable, str(path)],
            cwd=ROOT,
            env=env,
            text=True,
            capture_output=True,
            timeout=30,
        )
        assert completed.returncode == 0, f"{path.name}\nSTDOUT:\n{completed.stdout}\nSTDERR:\n{completed.stderr}"


def test_ci_runs_public_examples_explicitly():
    text = (ROOT / ".github/workflows/ci.yml").read_text()
    assert "Run public examples" in text
    assert "examples/[0-9][0-9]_*.py" in text


def test_sdist_manifest_includes_integrated_public_examples_and_current_contracts():
    text = (ROOT / "MANIFEST.in").read_text()
    assert "include PUBLIC-API-V0_1_0RC2.json" in text
    assert "include BACKEND-CONTRACT-V0_1_0RC2.json" in text
    assert "include docs/release/SOLVERPILOT-PUBLIC-DOCS-0.1.0RC2.md" in text
    assert "include docs/release/README-CHECKLIST-0.1.0RC2.md" in text
    assert "recursive-include examples *.py *.md" in text


def test_release_validator_requires_public_docs_and_all_examples():
    text = (ROOT / "tools/release_dist_manifest.py").read_text()
    assert 'f"docs/release/SOLVERPILOT-PUBLIC-DOCS-{CURRENT_RELEASE_DOC_VERSION}.md"' in text
    assert 'f"docs/release/README-CHECKLIST-{CURRENT_RELEASE_DOC_VERSION}.md"' in text
    for rel in [
        "docs/api/PUBLIC-API-v1.md",
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
        "docs/api/EXTENDED-MODELING-API.md",
        "TRACK-P-MERGE-PROVENANCE.json",
    ]:
        assert repr(rel) in text or f'"{rel}"' in text
