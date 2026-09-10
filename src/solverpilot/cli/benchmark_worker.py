from __future__ import annotations

import argparse
import json
import sys

from solverpilot.benchmark.worker import solve_file_worker


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--path", required=True)
    p.add_argument("--backend", required=True)
    p.add_argument("--portfolio", action="append", required=True)
    p.add_argument("--time-limit-s", type=float)
    p.add_argument("--solver-threads", type=int)
    args = p.parse_args()
    try:
        payload = solve_file_worker(
            path=args.path,
            backend=args.backend,
            portfolio=tuple(args.portfolio),
            time_limit_s=args.time_limit_s,
            solver_threads=args.solver_threads,
        )
        print(json.dumps(payload, sort_keys=True))
    except Exception as exc:
        print(json.dumps({"error_type": type(exc).__name__, "error": str(exc)}, sort_keys=True), file=sys.stderr)
        raise SystemExit(2) from exc


if __name__ == "__main__":
    main()
