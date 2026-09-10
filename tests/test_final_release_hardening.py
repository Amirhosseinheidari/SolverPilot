from __future__ import annotations

import ast
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

import solverpilot as sp
from solverpilot.cp.ortools_backend import ORToolsCPSATBackend
from solverpilot.problem.mps import MPSUnsupportedFeatureError, parse_mps


ROOT = Path(__file__).resolve().parents[1]


def test_production_source_contains_no_runtime_assert_statements():
    offenders: list[str] = []
    for path in sorted((ROOT / "src" / "solverpilot").rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.Assert):
                offenders.append(f"{path.relative_to(ROOT)}:{node.lineno}")
    assert offenders == []


def test_cp_controller_timeout_is_controlled_failure(monkeypatch):
    m = sp.CPModel()
    m.bool_var("x")
    problem = m.compile()
    backend = ORToolsCPSATBackend(controller_timeout_s=0.5)
    monkeypatch.setattr(ORToolsCPSATBackend, "_require", lambda self: None)

    def fake_run(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], timeout=kwargs["timeout"])

    monkeypatch.setattr(subprocess, "run", fake_run)
    with pytest.raises(RuntimeError, match="exceeded controller timeout"):
        backend.solve(problem)


def test_cp_invalid_json_response_is_controlled_failure(monkeypatch):
    m = sp.CPModel()
    m.bool_var("x")
    problem = m.compile()
    backend = ORToolsCPSATBackend()
    monkeypatch.setattr(ORToolsCPSATBackend, "_require", lambda self: None)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *a, **k: SimpleNamespace(returncode=0, stdout="not-json", stderr=""),
    )
    with pytest.raises(RuntimeError, match="invalid JSON"):
        backend.solve(problem)


def test_mps_multiple_ranges_vectors_fail_closed():
    text = """NAME X
ROWS
 N OBJ
 E R1
COLUMNS
 X R1 1
RHS
 RHS1 R1 1
RANGES
 RNG1 R1 1
 RNG2 R1 2
ENDATA
"""
    with pytest.raises(MPSUnsupportedFeatureError, match="multiple RANGES vectors"):
        parse_mps(text)


def test_mps_multiple_bounds_vectors_fail_closed():
    text = """NAME X
ROWS
 N OBJ
COLUMNS
 X OBJ 1
BOUNDS
 LO B1 X 0
 UP B2 X 1
ENDATA
"""
    with pytest.raises(MPSUnsupportedFeatureError, match="multiple BOUNDS vectors"):
        parse_mps(text)


def test_minlp_independent_global_proof_is_separate_from_solver_certification():
    from solverpilot.minlp.orchestrator import MINLPSolveResult

    r = MINLPSolveResult(
        status="proven_optimal",
        x=None,
        objective=None,
        lower_bound=None,
        gap=0.0,
        globally_proven=True,
        validation_valid=True,
        algorithm="test",
        iterations=(),
        raw_statistics={},
        proof_scope="solver_certified_under_convexity_assumptions",
    )
    assert r.globally_proven is True
    assert r.independently_verified_global is False


def test_release_workflow_has_full_optional_os_python_matrix_and_security_gate():
    text = (ROOT / ".github" / "workflows" / "release-qualification.yml").read_text(encoding="utf-8")
    native = text.split("  native-os-matrix:", 1)[1].split("  technical-gate:", 1)[0]
    assert 'os: [macos-latest, windows-latest]' in native
    assert 'python: ["3.12", "3.13", "3.14"]' in native
    assert "python-version: ${{ matrix.python }}" in native
    assert "static-correctness:" in text
    assert 'ruff==0.16.6' in text
    assert 'ruff check --select E9,F63,F7,F82' in text
    assert "security-sbom:" in text
    assert 'pip-audit==2.10.1' in text
    assert 'cyclonedx-json' in text
    gate = text.split("  technical-gate:", 1)[1]
    assert "security-sbom" in gate
    assert "STATIC_RESULT" in gate
    assert "SECURITY_RESULT" in gate


def test_attestation_workflow_is_current_and_least_privilege_bound():
    text = (ROOT / ".github" / "workflows" / "release-qualification.yml").read_text(encoding="utf-8")
    attest = text.split("  attest-dist:", 1)[1].split("  core-matrix:", 1)[0]
    assert "artifact-metadata: write" in attest
    assert "id-token: write" in attest
    assert "attestations: write" in attest
    assert "actions/attest@1e69f48acb82d1966a394da916b4c1698aa569d6 # v4.2.2" in attest
    assert "sbom-path: security-download/release-sbom.cdx.json" in attest


def test_research_pickle_cache_boundary_is_documented():
    security = (ROOT / "SECURITY.md").read_text(encoding="utf-8").lower()
    assert "pickle" in security
    assert "must never be" in security
    assert "untrusted" in security
