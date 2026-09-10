from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from statistics import median
from time import perf_counter

from solverpilot import read_mps, solve
from solverpilot.validate import PublicStatus


def main() -> None:
    ap = argparse.ArgumentParser(description="Run an externally supplied Netlib MPS instance without redistributing it")
    ap.add_argument("mps")
    ap.add_argument("--reference", type=float, required=True)
    ap.add_argument("--name", default=None)
    ap.add_argument("--repeats", type=int, default=7)
    ap.add_argument("--source-url", default=None)
    ap.add_argument("--reference-source-url", default=None)
    ap.add_argument("--output", default="benchmarks/results/m6-netlib-smoke.json")
    args = ap.parse_args()

    p = read_mps(args.mps)
    rows = []
    for backend in ("scipy-highs-ds", "scipy-highs-ipm"):
        walls = []
        result = None
        for _ in range(args.repeats):
            t0 = perf_counter()
            result = solve(p, backend=backend)
            walls.append(perf_counter() - t0)
            if result.status is not PublicStatus.VALID_OPTIMAL:
                raise RuntimeError(f"{backend} did not solve to validated optimum: {result.status}")
        assert result is not None
        abs_error = abs(float(result.objective) - args.reference)
        if abs_error > 1e-7 * max(1.0, abs(args.reference)):
            raise RuntimeError(f"{backend} objective {result.objective} disagrees with reference {args.reference}")
        rows.append({
            "backend": backend,
            "objective": result.objective,
            "reference": args.reference,
            "abs_error": abs_error,
            "median_wall_s": median(walls),
            "wall_s": walls,
        })
    file_sha256 = hashlib.sha256(Path(args.mps).read_bytes()).hexdigest()
    out = {
        "name": args.name or Path(args.mps).stem,
        "input_path": str(Path(args.mps).resolve()),
        "input_sha256": file_sha256,
        "source_url": args.source_url,
        "reference_source_url": args.reference_source_url,
        "n_variables": p.n_variables,
        "n_constraints": p.n_constraints,
        "nnz": p.nnz,
        "results": rows,
    }
    Path(args.output).write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
