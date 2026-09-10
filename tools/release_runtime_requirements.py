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
    ap.add_argument("--path", type=Path, help="Audit exact distributions in a wheel installation target")
    ns = ap.parse_args()
    names = ["numpy", "scipy"]
    if ns.extra:
        names.append(EXTRA_PACKAGE[ns.extra])
    if ns.path is not None:
        distributions = list(importlib.metadata.distributions(path=[str(ns.path)]))
        if not any(d.metadata["Name"].lower() == "solverpilot" for d in distributions):
            raise SystemExit("installation target must contain the SolverPilot wheel")
        lines = sorted(f"{d.metadata['Name']}=={d.version}" for d in distributions
                       if d.metadata['Name'].lower() != "solverpilot")
        if not lines:
            raise SystemExit("installation target has no runtime dependencies to audit")
    else:
        lines = [exact(name) for name in names]
    ns.output.write_text("\n".join(lines) + "\n")
    print(ns.output.read_text(encoding="utf-8"), end="")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
