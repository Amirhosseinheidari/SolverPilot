from __future__ import annotations

import argparse
import importlib.metadata
from pathlib import Path

EXTRA_PACKAGE = {
    "highs": "highspy",
    "osqp": "osqp",
    "scip": "pyscipopt",
    "nlopt": "nlopt",
    "casadi": "casadi",
    "conic": "casadi",
    "nlp": "casadi",
    "minlp": "casadi",
    "cp": "ortools",
}



def exact(name: str) -> str:
    return f"{name}=={importlib.metadata.version(name)}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--extra", choices=sorted(EXTRA_PACKAGE))
    ns = ap.parse_args()
    names = ["numpy", "scipy"]
    if ns.extra:
        names.append(EXTRA_PACKAGE[ns.extra])
    lines = [exact(name) for name in names]
    ns.output.write_text("\n".join(lines) + "\n")
    print(ns.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
