"""Fail-closed wheel identity checks for release-test Python processes.

Installed only into disposable CI environments, never into the product wheel.
A .pth bootstrap also runs in isolated (-I) children; an import finder checks
late sys.path changes. Receipts identify processes that actually imported the
package. This is a test invariant, not a sandbox against malicious Python code.
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.machinery
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import uuid
import zipfile

MODULE = "_solverpilot_artifact_guard"
_ACTIVE = False


def wheel_files(wheel: Path) -> dict[str, str]:
    with zipfile.ZipFile(wheel) as archive:
        files = {name: hashlib.sha256(archive.read(name)).hexdigest()
                 for name in archive.namelist()
                 if name.startswith("solverpilot/") and not name.endswith("/")}
    if "solverpilot/__init__.py" not in files:
        raise ValueError("wheel has no solverpilot package")
    return files


def install(py: Path, wheel: Path, receipts: Path) -> None:
    command = [str(py), "-I", "-c", "import sysconfig; print(sysconfig.get_path('purelib'))"]
    purelib = Path(subprocess.check_output(command, text=True).strip()).resolve()
    receipts = receipts.resolve()
    receipts.mkdir(parents=True, exist_ok=True)
    config = {"root": str(purelib), "receipts": str(receipts),
              "wheel_sha256": hashlib.sha256(wheel.read_bytes()).hexdigest(),
              "files": wheel_files(wheel)}
    shutil.copyfile(__file__, purelib / f"{MODULE}.py")
    (purelib / f"{MODULE}.json").write_text(json.dumps(config), encoding="utf-8")
    (purelib / "solverpilot_artifact_guard.pth").write_text(
        f"import {MODULE}; {MODULE}.activate()\n", encoding="utf-8")


def activate() -> None:
    # site.py normally prints .pth exceptions and continues. Identity failures
    # must terminate the process instead, including before pytest starts.
    global _ACTIVE
    if _ACTIVE:
        return
    _ACTIVE = True
    receipt = None
    state = {"schema": "solverpilot.release.import_identity.v1", "pid": os.getpid(),
             "executable": sys.executable, "imported": False, "passed": False}

    def save():
        if receipt is not None:
            receipt.write_text(json.dumps(state, sort_keys=True), encoding="utf-8")

    def fail(exc):
        state.update(passed=False, error=str(exc))
        try:
            save()
        finally:
            print(f"installed artifact identity failed: {exc}", file=sys.stderr, flush=True)
            os._exit(86)

    try:
        config = json.loads(Path(__file__).with_suffix(".json").read_text(encoding="utf-8"))
        root = Path(config["root"]).resolve()
        receipt = Path(config["receipts"]) / f"{os.getpid()}-{uuid.uuid4().hex}.json"
        expected = {str((root / name).resolve()): digest for name, digest in config["files"].items()}
        state.update(wheel_sha256=config["wheel_sha256"], package_root=str(root / "solverpilot"))
        for name, digest in expected.items():
            if hashlib.sha256(Path(name).read_bytes()).hexdigest() != digest:
                raise RuntimeError(f"wheel file differs: {name}")

        def check_spec(fullname, path=None, target=None):
            spec = importlib.machinery.PathFinder.find_spec(fullname, path, target)
            if spec is None or spec.origin is None:
                raise RuntimeError(f"missing installed module: {fullname}")
            origin = str(Path(spec.origin).resolve())
            if origin not in expected:
                raise RuntimeError(f"unexpected module origin: {fullname}: {origin}")
            if hashlib.sha256(Path(origin).read_bytes()).hexdigest() != expected[origin]:
                raise RuntimeError(f"module content differs: {fullname}")
            return spec

        check_spec("solverpilot")
        if any(name == "solverpilot" or name.startswith("solverpilot.") for name in sys.modules):
            raise RuntimeError("solverpilot imported before artifact guard")

        class Finder:
            def find_spec(self, fullname, path=None, target=None):
                if fullname != "solverpilot" and not fullname.startswith("solverpilot."):
                    return None
                try:
                    spec = check_spec(fullname, path, target)
                    if not state["imported"]:
                        state.update(imported=True, import_path=spec.origin)
                        save()
                    return spec
                except BaseException as exc:
                    fail(exc)

        sys.meta_path.insert(0, Finder())
        state["passed"] = True
        save()
    except BaseException as exc:
        fail(exc)


def report(receipts: Path, output: Path, *, minimum_imports: int = 1) -> dict:
    rows = [json.loads(path.read_text(encoding="utf-8")) for path in sorted(receipts.glob("*.json"))]
    imported = sum(bool(row["imported"]) for row in rows)
    passed = bool(rows) and all(row["passed"] for row in rows) and imported >= minimum_imports
    payload = {"schema": "solverpilot.release.import_identity_summary.v1",
               "processes": len(rows), "importing_processes": imported,
               "passed": passed, "receipts": rows}
    output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    if not passed:
        raise RuntimeError("installed artifact process identity checks failed")
    return payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("action", choices=("install", "report"))
    parser.add_argument("--wheel", type=Path)
    parser.add_argument("--install-wheel", action="store_true")
    parser.add_argument("--receipts", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    if args.action == "install":
        if args.wheel is None:
            parser.error("install requires --wheel")
        wheel = args.wheel
        if wheel.is_dir():
            wheels = list(wheel.glob("*.whl"))
            if len(wheels) != 1:
                parser.error("expected exactly one wheel")
            wheel = wheels[0]
        if args.install_wheel:
            subprocess.run([sys.executable, "-m", "pip", "install", "--force-reinstall",
                            "--no-deps", str(wheel.resolve())], check=True)
        install(Path(sys.executable), wheel, args.receipts)
    else:
        if args.output is None:
            parser.error("report requires --output")
        report(args.receipts, args.output)


if __name__ == "__main__":
    main()
