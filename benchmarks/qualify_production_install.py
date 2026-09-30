"""Qualify the same candidate wheel in two fresh, isolated local environments."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import zipfile

ROOT = Path(__file__).resolve().parents[1]
PILOT_FILES = ("README.md", "GUIDE-FA.md", "nd-production.json",
               "synthetic-minimum-commitments.json", "feedback-template.md",
               "feedback-template.json", "LICENSE-ND-PYOMO-CODE.txt",
               "COMPARISON-PROTOCOL.md", "comparison-feedback-template.json")


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def clean_env():
    env = dict(os.environ)
    for key in ("PYTHONPATH", "PYTHONHOME", "PYTHONSTARTUP", "PYTHONUSERBASE"):
        env.pop(key, None)
    env["PYTHONNOUSERSITE"] = "1"
    env["PIP_DISABLE_PIP_VERSION_CHECK"] = "1"
    return env


def _run(command, work, label, *, timeout=600):
    print(label, flush=True)
    result = subprocess.run([str(item) for item in command], cwd=work, env=clean_env(),
                            capture_output=True, text=True, encoding="utf-8", errors="replace", timeout=timeout)
    (work / "logs" / (label + ".log")).write_text(result.stdout + "\n" + result.stderr, encoding="utf-8")
    if result.returncode:
        raise RuntimeError(f"{label} failed; inspect {work / 'logs' / (label + '.log')}")
    return result.stdout


def _python(env_path):
    return env_path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def copy_pilot_docs(destination):
    """Copy shipped kit inputs only, never local responses or nested outputs."""
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=False)
    for name in PILOT_FILES:
        shutil.copy2(ROOT / "docs/pilot" / name, destination / name)


def installed_identity(py, work, label):
    code = """import json, pathlib, sys, solverpilot
from solverpilot._identity import source_tree_sha256
root=pathlib.Path(sys.prefix).resolve()
package=pathlib.Path(solverpilot.__file__).resolve()
assert package.is_relative_to(root), 'SolverPilot imported outside the clean environment'
print(json.dumps({'isolated':sys.flags.isolated == 1, 'installed_inside_environment':True,
 'package_relative_path':str(package.relative_to(root)), 'version':solverpilot.__version__,
 'source_sha256':source_tree_sha256()}))
"""
    return json.loads(_run([py, "-I", "-c", code], work, label))


def qualify_install(wheel, output, *, repeats=6, include_model=False):
    if not include_model:
        raise ValueError("full local model export requires explicit --include-model")
    if isinstance(repeats, bool) or not isinstance(repeats, int) or not 2 <= repeats <= 100 or repeats % 2:
        raise ValueError("repeats must be an even integer between 2 and 100")
    wheel, output = Path(wheel).resolve(), Path(output).resolve()
    if not wheel.is_file() or wheel.suffix != ".whl" or not wheel.name.startswith("solverpilot-"):
        raise ValueError("supply the candidate solverpilot wheel")
    if output.exists():
        raise FileExistsError("refusing to overwrite qualification evidence")
    wheel_sha = digest(wheel)
    output.mkdir(parents=True)
    (output / "logs").mkdir()
    first, second = output / "origin-env", output / "replay-env"
    _run([sys.executable, "-I", "-m", "venv", first], output, "create-origin-environment")
    py_a = _python(first)
    _run([py_a, "-I", "-m", "pip", "install", wheel], output, "install-origin-wheel")
    packages = json.loads(_run([py_a, "-I", "-m", "pip", "list", "--format=json"], output, "freeze-origin-packages"))
    pins = sorted(f"{item['name']}=={item['version']}" for item in packages
                  if item["name"].lower() not in {"pip", "setuptools", "wheel", "solverpilot"})
    lock = output / "runtime-lock.txt"
    lock.write_text("\n".join(pins) + "\n", encoding="utf-8")
    origin = installed_identity(py_a, output, "verify-origin-import")
    _run([sys.executable, "-I", "-m", "venv", second], output, "create-replay-environment")
    py_b = _python(second)
    _run([py_b, "-I", "-m", "pip", "install", "-r", lock, wheel], output, "install-replay-wheel")
    replay_identity = installed_identity(py_b, output, "verify-replay-import")
    runner = ROOT / "benchmarks/qualify_production_workflow.py"
    records = []
    for name in ("nd-production", "synthetic-minimum-commitments"):
        fixture = ROOT / "docs/pilot" / (name + ".json")
        directory = output / name
        _run([py_a, "-I", runner, "run", "--input", fixture, "--output", directory,
              "--repeats", repeats, "--include-model"], output, name + "-origin")
        replay_path = output / (name + "-clean-replay.json")
        _run([py_b, "-I", runner, "replay", "--study", directory / "study.json", "--output", replay_path],
             output, name + "-replay")
        qualification = json.loads((directory / "qualification.json").read_text(encoding="utf-8"))
        replay = json.loads(replay_path.read_text(encoding="utf-8"))
        records.append({"fixture": name, "fixture_sha256": digest(fixture),
            "technical_gate_passed": qualification["technical_gate_passed"],
            "clean_replay_passed": replay["passed"],
            "environment_matches": qualification["environment"]["packages"] == replay["environment"]["packages"],
            "study_source_matches_wheel": qualification["source_sha256"] == origin["source_sha256"],
            "cases": len(replay["scenarios"]),
            "qualification_sha256": digest(directory / "qualification.json"),
            "replay_sha256": digest(replay_path)})
    identity_ok = bool(origin["isolated"] and replay_identity["isolated"]
                       and origin["source_sha256"] == replay_identity["source_sha256"]
                       and digest(wheel) == wheel_sha)
    passed = identity_ok and all(row["technical_gate_passed"] and row["clean_replay_passed"]
                                and row["environment_matches"] and row["study_source_matches_wheel"] for row in records)
    evidence = {"schema": "solverpilot.production-pilot-installation.v1", "passed": bool(passed),
        "wheel": wheel.name, "wheel_sha256": wheel_sha, "runner_sha256": digest(runner),
        "runtime_lock_sha256": digest(lock), "origin": origin, "replay": replay_identity, "fixtures": records,
        "scope": "same candidate wheel in two fresh local environments and separate Python processes",
        "source_attestation": False, "human_pilot_completed": False, "real_user_data_qualified": False}
    (output / "qualification.json").write_text(json.dumps(evidence, indent=2), encoding="utf-8")
    # Portable public pilot kit: no virtual environments, credentials, machine logs or user feedback.
    kit = output / "portable-pilot"
    (kit / "benchmarks").mkdir(parents=True)
    (kit / "docs").mkdir()
    copy_pilot_docs(kit / "docs/pilot")
    shutil.copy2(ROOT / "docs/PRODUCTION-EVIDENCE-WORKFLOW.md", kit / "docs")
    shutil.copy2(runner, kit / "benchmarks")
    shutil.copy2(wheel, kit)
    shutil.copy2(lock, kit)
    shutil.copy2(output / "qualification.json", kit / "installation-evidence.json")
    contents = {str(path.relative_to(kit).as_posix()): digest(path)
                for path in sorted(kit.rglob("*")) if path.is_file()}
    (kit / "SHA256SUMS.json").write_text(json.dumps(contents, indent=2), encoding="utf-8")
    with zipfile.ZipFile(output / "portable-pilot.zip", "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(kit.rglob("*")):
            if path.is_file():
                archive.write(path, path.relative_to(kit))
    print(json.dumps({"passed": bool(passed), "output": str(output)}), flush=True)
    return evidence


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wheel", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--include-model", action="store_true")
    args = parser.parse_args()
    return 0 if qualify_install(args.wheel, args.output, repeats=args.repeats, include_model=args.include_model)["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
