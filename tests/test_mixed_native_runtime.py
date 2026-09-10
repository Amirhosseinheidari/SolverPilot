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
