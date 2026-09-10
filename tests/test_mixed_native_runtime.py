"""Native integrations must coexist without terminating the Python process."""
import subprocess
import sys

import pytest
import numpy as np


@pytest.mark.native
def test_mixed_native_health_probes_survive():
    pytest.importorskip("casadi")
    pytest.importorskip("highspy")
    program = '''
from solverpilot import builtin_backend_candidates
from solverpilot.backends import probe_backend
for backend in builtin_backend_candidates():
    print("PROBING", backend.manifest.name, flush=True)
    report = probe_backend(backend)
    print("RESULT", report.status.value, flush=True)
    if report.available and not report.healthy:
        raise RuntimeError(str(report))
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=90)
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.native
def test_default_solve_with_native_and_bundled_highs_installed():
    pytest.importorskip("casadi")
    pytest.importorskip("highspy")
    program = '''
from solverpilot import LinearProblem, solve
p = LinearProblem.from_data(A=[[1.]], c=[1.], variable_lower=[0.],
    variable_upper=[2.], constraint_lower=[1.], constraint_upper=[2.])
r = solve(p)
assert r.validation.valid and abs(r.objective - 1.) < 1e-8, r
'''
    result = subprocess.run([sys.executable, "-c", program], capture_output=True, text=True, timeout=60)
    assert result.returncode == 0, result.stdout + result.stderr


def test_highspy_unloadable_extension_is_unavailable(monkeypatch):
    from solverpilot.backends import HighspyNativeBackend
    import importlib
    monkeypatch.setattr(importlib.util, "find_spec", lambda name: object())
    def broken_import(name):
        raise ImportError("undefined native symbol")
    monkeypatch.setattr(importlib, "import_module", broken_import)
    assert not HighspyNativeBackend().is_available()


def test_casadi_highs_worker_crash_is_a_controlled_backend_error(monkeypatch):
    from types import SimpleNamespace
    from solverpilot import LinearProblem, execute, PublicStatus
    from solverpilot.backends import CasadiHighsBridgeBackend
    backend = CasadiHighsBridgeBackend()
    monkeypatch.setattr(CasadiHighsBridgeBackend, "is_available", lambda self: True)
    monkeypatch.setattr(subprocess, "run", lambda *a, **k: SimpleNamespace(
        returncode=-11, stdout=b"", stderr=b"native worker crashed"))
    problem = LinearProblem.from_data(A=np.zeros((0, 1)), c=[1.], variable_lower=[0.],
        variable_upper=[1.], constraint_lower=[], constraint_upper=[])
    result = execute(problem, backend)
    assert result.status is PublicStatus.ERROR
    assert result.raw_statistics["worker_returncode"] == -11
    assert result.raw_statistics["process_isolated"] is True


def test_casadi_highs_worker_timeout_is_a_controlled_backend_error(monkeypatch):
    from solverpilot import LinearProblem
    from solverpilot.backends import CasadiHighsBridgeBackend
    backend = CasadiHighsBridgeBackend(time_limit_s=1.)
    monkeypatch.setattr(CasadiHighsBridgeBackend, "is_available", lambda self: True)
    def timeout(*args, **kwargs):
        raise subprocess.TimeoutExpired(args[0], kwargs["timeout"])
    monkeypatch.setattr(subprocess, "run", timeout)
    problem = LinearProblem.from_data(A=np.zeros((0, 1)), c=[1.], variable_lower=[0.],
        variable_upper=[1.], constraint_lower=[], constraint_upper=[])
    result = backend.solve(problem)
    assert result.backend_status == "solver_error" and result.x is None
    assert "timeout" in result.raw_statistics["error"]
