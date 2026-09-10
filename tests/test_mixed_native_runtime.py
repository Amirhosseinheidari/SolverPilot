"""Native integrations must coexist without terminating the Python process."""
import subprocess
import sys

import pytest


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
