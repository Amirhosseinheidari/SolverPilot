from __future__ import annotations

import ast
import inspect
import json
from pathlib import Path
import tomllib

import yaml
from packaging.version import Version

import solverpilot
from solverpilot import builtin_backend_candidates, default_registry

ROOT = Path(__file__).resolve().parents[1]


def _historical_name(name: str) -> str:
    return "SolverPilotError" if name == "OptiMindError" else name


def _historical_module(module: str | None) -> str | None:
    return module.replace("optimind.", "solverpilot.") if module else None


def _historical_qualname(name: str | None) -> str | None:
    return "SolverPilotError" if name == "OptiMindError" else name


def _signature(obj):
    try:
        return str(inspect.signature(obj))
    except (TypeError, ValueError):
        return None


def test_m31_version_is_pep440_rc_and_coherent():
    snapshot_version = json.loads((ROOT / "docs/history/frozen-m31/PUBLIC-API-M31.json").read_text())["package_version"]
    assert snapshot_version == "0.0.36rc1"
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["version"] == solverpilot.__version__
    assert Version(project["version"]) >= Version(snapshot_version)


def test_m31_public_api_snapshot_exactly_matches_runtime():
    payload = json.loads((ROOT / "docs/history/frozen-m31/PUBLIC-API-M31.json").read_text())
    assert payload["package_version"] == "0.0.36rc1"
    assert [_historical_name(row["name"]) for row in payload["symbols"]] == solverpilot.__all__
    for row in payload["symbols"]:
        obj = getattr(solverpilot, _historical_name(row["name"]))
        assert _historical_module(row["module"]) == getattr(obj, "__module__", None)
        assert _historical_qualname(row["qualname"]) == getattr(obj, "__qualname__", None)
        assert row["signature"] == _signature(obj)


def test_m31_backend_contract_matches_runtime_ids():
    payload = json.loads((ROOT / "docs/history/frozen-m31/BACKEND-CONTRACT-M31.json").read_text())
    groups = payload["stable_backend_ids"]
    expected = {name for names in groups.values() for name in names}
    actual = {b.manifest.name for b in builtin_backend_candidates()}
    assert expected == actual
    assert set(groups["verification_only"]).isdisjoint(default_registry().names())
    assert payload["production_baseline_policy"]["learned_lp_performance_routing"] is False


def test_all_package_sources_parse_with_python310_grammar():
    errors = []
    for path in sorted((ROOT / "src").rglob("*.py")):
        try:
            ast.parse(path.read_text(encoding="utf-8"), filename=str(path), feature_version=(3, 10))
        except SyntaxError as exc:
            errors.append((str(path), exc.lineno, exc.msg))
    assert not errors


def test_ci_matrix_covers_declared_python_floor_and_three_desktop_oses():
    workflow = yaml.safe_load((ROOT / "docs/history/frozen-workflows/m31-compatibility.yml").read_text())
    jobs = workflow["jobs"]
    core = jobs["core-matrix"]["strategy"]["matrix"]
    assert core["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert core["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]
    native_py = jobs["native-python-matrix"]["strategy"]["matrix"]
    assert native_py["python"] == ["3.10", "3.11", "3.12", "3.13"]
    assert set(native_py["extra"]) == {"highs", "osqp", "scip", "nlopt", "casadi"}
    native_os = jobs["native-os-matrix"]["strategy"]["matrix"]
    assert native_os["os"] == ["ubuntu-latest", "macos-latest", "windows-latest"]


def test_publication_owner_fields_remain_uninvented():
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    assert project["license"] == "Apache-2.0"
    assert project["license-files"] == ["LICENSE"]
    assert project["authors"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert project["maintainers"] == [{"name": "Amirhossein Heidari Rashtabad"}]
    assert "urls" not in project


def test_m31_compatibility_matrix_never_promotes_unexecuted_cells():
    payload = json.loads((ROOT / "docs/history/frozen-m31/M31-COMPATIBILITY-MATRIX.json").read_text())
    assert payload["schema"] == "optimind.m31.compatibility_matrix.v1"
    executed = [x for x in payload["core_matrix"] if x["status"] == "executed_pass"]
    assert executed == [{"os": "linux", "python": "3.13", "status": "executed_pass", "evidence": "verification/2026-09-06/m31-linux-py313-smoke.json"}]
    assert all(x["status"] in {"executed_pass", "pending_ci"} for x in payload["core_matrix"])
    assert payload["public_1_0_compatibility_gate"] is False


def test_clean_pythonpath_filters_hook_paths_with_windows_or_posix_separators(monkeypatch, tmp_path):
    import solverpilot.benchmark.runner as runner

    good = tmp_path / "good"
    good.mkdir()
    posix_blocked = tmp_path / "python-hooks" / "x"
    windows_blocked = r"C:\\sandbox\\pyvenv-overrides\\x"
    monkeypatch.setattr(runner.sys, "path", [str(good), str(posix_blocked), windows_blocked])
    cleaned = runner._clean_pythonpath().split(runner.os.pathsep)
    assert str(good.resolve()) in cleaned
    assert not any("python-hooks" in item.replace("\\", "/") for item in cleaned)
    assert not any("pyvenv-overrides" in item.replace("\\", "/") for item in cleaned)
