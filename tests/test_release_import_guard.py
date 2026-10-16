"""Behavioral tests using disposable interpreters and wheel-shaped fixtures."""
import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys
import zipfile

import pytest


@pytest.fixture
def guarded(tmp_path):
    source = Path(__file__).resolve().parents[1] / "tools/release_import_guard.py"
    spec = importlib.util.spec_from_file_location("guard_under_test", source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    venv = tmp_path / "venv"
    subprocess.run([sys.executable, "-I", "-m", "venv", "--without-pip", str(venv)], check=True)
    py = venv / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    purelib = Path(subprocess.check_output(
        [str(py), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"], text=True).strip())
    wheel = tmp_path / "fixture.whl"
    with zipfile.ZipFile(wheel, "w") as archive:
        archive.writestr("solverpilot/__init__.py", "VALUE = 17\n")
        archive.writestr("solverpilot/child.py", "VALUE = 19\n")
        archive.writestr("solverpilot/data.json", '{"value": 23}')
    with zipfile.ZipFile(wheel) as archive:
        archive.extractall(purelib)
    receipts = tmp_path / "receipts"
    module.install(py, wheel, receipts)
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME"):
        env.pop(key, None)
    return module, py, purelib, receipts, env


@pytest.mark.parametrize("flags", [[], ["-I"]])
def test_parent_and_isolated_child_import_exact_installed_files(guarded, tmp_path, flags):
    module, py, purelib, receipts, env = guarded
    code = ("import solverpilot, solverpilot.child, subprocess, sys; "
            "assert solverpilot.VALUE == 17 and solverpilot.child.VALUE == 19; "
            "subprocess.run([sys.executable, '-I', '-c', 'import solverpilot'], check=True)")
    result = subprocess.run([str(py), *flags, "-c", code], env=env, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    payload = module.report(receipts, tmp_path / "summary.json", minimum_imports=2)
    assert payload["importing_processes"] == 2
    assert all(Path(row["import_path"]).is_relative_to(purelib)
               for row in payload["receipts"] if row["imported"])


@pytest.mark.parametrize("late", [False, True])
def test_source_shadowing_fails_even_after_startup(guarded, tmp_path, late):
    module, py, _, receipts, env = guarded
    source = tmp_path / "src"
    (source / "solverpilot").mkdir(parents=True)
    (source / "solverpilot/__init__.py").write_text("VALUE = 999\n")
    if late:
        args = ["-I", "-c", f"import sys; sys.path.insert(0, {str(source)!r}); import solverpilot"]
    else:
        env["PYTHONPATH"] = str(source)
        args = ["-c", "import solverpilot"]
    result = subprocess.run([str(py), *args], env=env, capture_output=True, text=True)
    assert result.returncode == 86
    assert "unexpected module origin" in result.stderr
    with pytest.raises(RuntimeError, match="identity checks failed"):
        module.report(receipts, tmp_path / "summary.json")


@pytest.mark.parametrize("name", ["child.py", "data.json"])
def test_changed_installed_module_or_data_is_rejected(guarded, name):
    _, py, purelib, receipts, env = guarded
    (purelib / "solverpilot" / name).write_text("changed")
    result = subprocess.run([str(py), "-I", "-c", "import solverpilot"], env=env,
                            capture_output=True, text=True)
    assert result.returncode == 86 and "wheel file differs" in result.stderr
    assert any(not json.loads(p.read_text())["passed"] for p in receipts.glob("*.json"))


def test_process_that_never_imported_package_is_not_verification(guarded, tmp_path):
    module, py, _, receipts, env = guarded
    subprocess.run([str(py), "-I", "-c", "pass"], env=env, check=True)
    with pytest.raises(RuntimeError, match="identity checks failed"):
        module.report(receipts, tmp_path / "summary.json")
