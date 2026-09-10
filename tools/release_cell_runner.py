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


def run(cmd: list[str], *, stdout=None, stderr=None, cwd: Path | None = None, env: dict[str, str] | None = None) -> None:
    subprocess.run(cmd, check=True, stdout=stdout, stderr=stderr, cwd=cwd, env=env)


def clean_runtime_env() -> dict[str, str]:
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    env["PYTHONNOUSERSITE"] = "1"
    return env


def run_installed_public_examples(py: Path, *, work: Path, prefix: str) -> Path:
    env = clean_runtime_env()
    rows: list[dict[str, object]] = []
    for example in sorted((ROOT / "examples").glob("[0-9][0-9]_*.py")):
        proc = subprocess.run(
            [str(py), str(example.resolve())],
            cwd=work,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=90,
        )
        rows.append({
            "example": example.name,
            "returncode": proc.returncode,
            "stdout": proc.stdout[-4000:],
            "stderr": proc.stderr[-4000:],
            "passed": proc.returncode == 0,
        })
        if proc.returncode != 0:
            raise SystemExit(f"installed-wheel example failed: {example.name}\n{proc.stdout}\n{proc.stderr}")
    if len(rows) != 14:
        raise SystemExit(f"expected 14 public examples, got {len(rows)}")
    out = work / f"{prefix}-installed-examples.json"
    out.write_text(json.dumps({
        "schema": "solverpilot.release.installed_examples.v1",
        "count": len(rows),
        "passed": all(bool(row["passed"]) for row in rows),
        "examples": rows,
    }, indent=2, sort_keys=True) + "\n")
    return out


EXTRA_FOCUSED_TESTS = {
    "conic": ["tests/test_p6_conic_ir.py", "tests/test_p6_semantic_cones.py", "tests/test_p6_conformance_and_schema.py"],
    "nlp": ["tests/test_p7_nlp_ir_ad.py", "tests/test_p7_ipopt_backend.py", "tests/test_p7_conformance.py"],
    "minlp": ["tests/test_p8_minlp.py"],
    "cp": ["tests/test_p9_cp_core.py", "tests/test_p9_ortools_static_contract.py"],
}

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--prefix", required=True)
    ap.add_argument(
        "--extra",
        choices=["highs", "osqp", "scip", "nlopt", "casadi", "conic", "nlp", "minlp", "cp"],
    )
    ap.add_argument("--minimum-deps", action="store_true")
    ap.add_argument("--skip-regression", action="store_true")
    ns = ap.parse_args()

    dist = ns.dist.resolve()
    wheels = sorted(dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {dist}")
    wheel = wheels[0]
    project = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))["project"]

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
        run([str(py), str(ROOT / "tools/release_resolver_report.py")], stdout=f, cwd=ROOT)

    smoke_path = work / f"{ns.prefix}-smoke.json"
    smoke_cmd = [str(py), str(ROOT / "tools/release_smoke.py"), "--forbid-source-root", str(ROOT)]
    if ns.extra:
        smoke_cmd += ["--require-extra", ns.extra]
    with smoke_path.open("w") as f:
        run(smoke_cmd, stdout=f, cwd=ROOT, env=clean_runtime_env())

    requirements_path = work / f"{ns.prefix}-runtime-requirements.txt"
    req_cmd = [str(py), str(ROOT / "tools/release_runtime_requirements.py"), "--output", str(requirements_path)]
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
        "schema": "solverpilot.release.runtime_dependency_audit.v1",
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
        resolved = json.loads(resolver_path.read_text(encoding="utf-8"))
        packages = resolved["packages"]
        if packages.get("numpy") != MIN_NUMPY or packages.get("scipy") != MIN_SCIPY:
            raise SystemExit(f"minimum dependency drift: {packages}")

    focused = EXTRA_FOCUSED_TESTS.get(ns.extra, [])
    if focused:
        focused_junit = work / f"{ns.prefix}-extra-pytest.xml"
        run([str(py), "-m", "pytest", *focused, f"--junitxml={focused_junit}"], cwd=ROOT, env=clean_runtime_env())

    if not ns.skip_regression:
        junit = work / f"{ns.prefix}-pytest.xml"
        run([str(py), "-m", "pytest", f"--junitxml={junit}"], cwd=ROOT, env=clean_runtime_env())

    installed_examples = run_installed_public_examples(py, work=work, prefix=ns.prefix)

    summary = {
        "schema": "solverpilot.release.cell_runner.v1",
        "prefix": ns.prefix,
        "extra": ns.extra,
        "minimum_deps": ns.minimum_deps,
        "wheel": wheel.name,
        "installed_examples": installed_examples.name,
        "passed": True,
    }
    (work / f"{ns.prefix}-cell-summary.json").write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
