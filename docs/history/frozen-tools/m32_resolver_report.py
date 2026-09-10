from __future__ import annotations

import importlib.metadata
import json
import platform
import subprocess
import sys

PACKAGES = ["solverpilot", "numpy", "scipy", "pip", "highspy", "osqp", "pyscipopt", "nlopt", "casadi"]


def version(name: str):
    try:
        return importlib.metadata.version(name)
    except importlib.metadata.PackageNotFoundError:
        return None

payload = {
    "schema": "solverpilot.m32.resolver_report.v1",
    "python": platform.python_version(),
    "python_implementation": platform.python_implementation(),
    "platform": platform.platform(),
    "machine": platform.machine(),
    "packages": {name: version(name) for name in PACKAGES},
}
check = subprocess.run([sys.executable, "-m", "pip", "check"], capture_output=True, text=True)
payload["pip_check"] = {"returncode": check.returncode, "stdout": check.stdout, "stderr": check.stderr}
print(json.dumps(payload, indent=2, sort_keys=True))
raise SystemExit(0 if check.returncode == 0 else 2)
