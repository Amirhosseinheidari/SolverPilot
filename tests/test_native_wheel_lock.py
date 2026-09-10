import hashlib
import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
LOCK = ROOT / "benchmarks" / "native-wheel-lock.json"
TOOL = ROOT / "tools" / "verify_native_wheels.py"


def _load_tool():
    spec = importlib.util.spec_from_file_location("verify_native_wheels", TOOL)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_native_wheel_lock_is_well_formed():
    tool = _load_tool()
    payload = tool.load_lock(LOCK)
    assert set(payload["packages"]) == {"highspy", "osqp", "pyscipopt", "nlopt"}
    for item in payload["packages"].values():
        assert item["url"].startswith("https://files.pythonhosted.org/")
        assert item["filename"].endswith(".whl")
        assert len(item["sha256"]) == 64


def test_sha256_file(tmp_path):
    tool = _load_tool()
    path = tmp_path / "x.whl"
    data = b"solverpilot-wheel-integrity-test"
    path.write_bytes(data)
    assert tool.sha256_file(path) == hashlib.sha256(data).hexdigest()


def test_verify_wheel_dir_reports_missing(tmp_path):
    tool = _load_tool()
    payload = tool.load_lock(LOCK)
    results = tool.verify_wheel_dir(payload, tmp_path)
    assert all(not item["ok"] and item["reason"] == "missing" for item in results.values())
