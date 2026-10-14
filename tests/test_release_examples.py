"""Behavioral coverage for installed-wheel public example discovery and execution."""

import importlib.util
import json
from pathlib import Path
import subprocess

import pytest


@pytest.fixture
def runner():
    path = Path(__file__).resolve().parents[1] / "tools/release_cell_runner.py"
    spec = importlib.util.spec_from_file_location("release_examples_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


@pytest.mark.parametrize("names, message", [
    ([], "nonempty"),
    (["02_second.py"], "contiguous"),
    (["01_first.py", "03_third.py"], "contiguous"),
    (["01_first.py", "01_duplicate.py"], "duplicate"),
    (["00_zero.py", "01_first.py"], "invalid"),
    (["1_unpadded.py"], "invalid"),
    (["001_overpadded.py"], "invalid"),
    (["01a_malformed.py"], "invalid"),
])
def test_invalid_sequence_fails_before_execution(runner, tmp_path, monkeypatch, names, message):
    examples = tmp_path / "examples"
    examples.mkdir()
    for name in names:
        (examples / name).touch()
    monkeypatch.setattr(runner, "ROOT", tmp_path)

    def unexpected_run(*args, **kwargs):
        pytest.fail("invalid example inventory must fail before launching Python")

    monkeypatch.setattr(runner.subprocess, "run", unexpected_run)
    with pytest.raises(SystemExit, match=message):
        runner.run_installed_public_examples(Path("python"), work=tmp_path, prefix="test")


def test_all_contiguous_examples_execute_in_numeric_order(runner, tmp_path, monkeypatch):
    examples = tmp_path / "examples"
    examples.mkdir()
    expected = [f"{number:02d}_example.py" for number in range(1, 101)]
    for name in reversed(expected):
        (examples / name).touch()
    (examples / "helper.py").touch()
    calls = []

    def fake_run(cmd, **kwargs):
        calls.append((cmd, kwargs))
        return subprocess.CompletedProcess(cmd, 0, "ok", "")

    monkeypatch.setattr(runner, "ROOT", tmp_path)
    monkeypatch.setattr(runner.subprocess, "run", fake_run)
    monkeypatch.setenv("PYTHONPATH", "unwanted-source-import-path")
    output = runner.run_installed_public_examples(Path("python"), work=tmp_path, prefix="test")
    payload = json.loads(output.read_text())
    assert [Path(cmd[1]).name for cmd, _ in calls] == expected
    assert all(kwargs["cwd"] == tmp_path and "PYTHONPATH" not in kwargs["env"]
               and kwargs["env"]["PYTHONNOUSERSITE"] == "1" for _, kwargs in calls)
    assert payload["count"] == 100 and payload["passed"]
    assert [row["example"] for row in payload["examples"]] == expected


def test_repository_examples_form_a_complete_sequence(runner):
    examples = runner.public_examples(runner.ROOT / "examples")
    assert set(examples) == set((runner.ROOT / "examples").glob("[0-9]*_*.py"))
