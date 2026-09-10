from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PIP_AUDIT_VERSION = "2.10.1"
MIN_NUMPY = "2.2.0"
MIN_SCIPY = "1.15.0"


def venv_python(venv: Path) -> Path:
    if os.name == "nt":
        return venv / "Scripts" / "python.exe"
    return venv / "bin" / "python"


def run(cmd: list[str], *, stdout=None, stderr=None, cwd: Path | None = None) -> None:
    subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, cwd=cwd)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument("--extra", choices=["highs", "osqp", "scip", "nlopt", "casadi"])
    ap.add_argument("--minimum-deps", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ns = ap.parse_args()

    dist = ns.dist.resolve()
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist}")
    wheel = wheels[0]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]

    work = Path.cwd()
    runtime_env = work / f".{ns.prefix}-runtime-venv"
    audit_env = work / f".{ns.prefix}-audit-venv"
    for path in (runtime_env, audit_env):
        if path.exists():
            shutil.rmtree(path)

    run([sys.executable, "-m", "venv", str(runtime_env)])
    py = venv_python(runtime_env)
    if ns.minimum_deps:
        run([str(py), "-m", "pip", "install", f"numpy=={MIN_NUMPY}", f"scipy=={MIN_SCIPY}"])

    install_args = [str(wheel)] + list(project["optional-dependencies"]["test"])
    if ns.extra:
        install_args.extend(project["optional-dependencies"][ns.extra])
    run([str(py), "-m", "pip", "install", *install_args])

    # Dependency consistency / environment evidence.
    resolver_path = work / f"{ns.prefix}-resolver.json"
    with resolver_path.open("w") as f:
        run([str(py), str(ROOT / "tools/m33_resolver_report.py")], stdout=f, cwd=ROOT)

    smoke_path = work / f"{ns.prefix}-smoke.json"
    smoke_cmd = [str(py), str(ROOT / "tools/m33_release_smoke.py"), "--forbid-source-root", str(ROOT)]
    if ns.extra:
        smoke_cmd += ["--require-extra", ns.extra]
    with smoke_path.open("w") as f:
        run(smoke_cmd, stdout=f, cwd=ROOT)

    requirements_path = work / f"{ns.prefix}-runtime-requirements.txt"
    req_cmd = [str(py), str(ROOT / "tools/m33_runtime_requirements.py"), "--output", str(requirements_path)]
    if ns.extra:
        req_cmd += ["--extra", ns.extra]
    run(req_cmd, cwd=ROOT)

    # Security audit tool is isolated from the runtime environment.
    run([sys.executable, "-m", "venv", str(audit_env)])
    audit_py = venv_python(audit_env)
    run([str(audit_py), "-m", "pip", "install", f"pip-audit=={PIP_AUDIT_VERSION}"])
    sbom_path = work / f"{ns.prefix}-sbom.cyclonedx.json"
    audit_stdout = work / f"{ns.prefix}-pip-audit.stdout.txt"
    audit_stderr = work / f"{ns.prefix}-pip-audit.stderr.txt"
    with audit_stdout.open("w") as out, audit_stderr.open("w") as err:
        proc = subprocess.run(
            [str(audit_py), "-m", "pip_audit", "-r", str(requirements_path), "--format", "cyclonedx-json", "--output", str(sbom_path)],
            stdout=out,
            stderr=err,
            cwd=ROOT,
        )
    audit_status = {
        "schema": "solverpilot.m33.runtime_dependency_audit.v1",
        "returncode": proc.returncode,
        "requirements": requirements_path.name,
        "sbom": sbom_path.name,
        "passed": proc.returncode == 0,
        "ignore_list": [],
    }
    (work / f"{ns.prefix}-pip-audit-status.json").write_text(json.dumps(audit_status, indent=2, sort_keys=True) + "\n")
    if proc.returncode != 0:
        raise SystemExit(proc.returncode)

    if ns.minimum_deps:
        resolved = json.loads(resolver_path.read_text())
        packages = resolved["packages"]
        if packages.get("numpy") != MIN_NUMPY or packages.get("scipy") != MIN_SCIPY:
            raise SystemExit(f"minimum dependency drift: {packages}")

    if not ns.skip_regression:
        junit = work / f"{ns.prefix}-pytest.xml"
        run([str(py), "-m", "pytest", f"--junitxml={junit}"], cwd=ROOT)

    summary = {
        "schema": "solverpilot.m33.cell_runner.v1",
        "prefix": ns.prefix,
        "extra": ns.extra,
        "minimum_deps": ns.minimum_deps,
        "wheel": wheel.name,
        "passed": True,
    }
    (work / f"{ns.prefix}-cell-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
