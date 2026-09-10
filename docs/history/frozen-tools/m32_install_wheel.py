from __future__ import annotations

import argparse
import subprocess
import sys
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dist", type=Path, required=True)
    ap.add_argument("--test", action="store_true")
    ap.add_argument("--extra", choices=["highs", "osqp", "scip", "nlopt", "casadi"])
    ns = ap.parse_args()
    wheels = sorted(ns.dist.glob("*.whl"))
    if len(wheels) != 1:
        raise SystemExit(f"expected exactly one wheel in {ns.dist}")
    project = tomllib.loads((ROOT / "pyproject.toml").read_text())["project"]
    deps = [str(wheels[0])]
    if ns.test:
        deps.extend(project["optional-dependencies"]["test"])
    if ns.extra:
        deps.extend(project["optional-dependencies"][ns.extra])
    subprocess.run([sys.executable, "-m", "pip", "install", *deps], check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
