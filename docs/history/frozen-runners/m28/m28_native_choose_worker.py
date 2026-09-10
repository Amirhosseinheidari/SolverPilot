from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path
from time import perf_counter

from optimind import execute
from optimind.backends import BundledHighsCAPIBackend
from optimind.validate import PublicStatus

TERMINAL = {PublicStatus.VALID_OPTIMAL, PublicStatus.INFEASIBLE, PublicStatus.UNBOUNDED}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--problem', type=Path, required=True)
    ap.add_argument('--solver', choices=('choose', 'simplex', 'ipm'), required=True)
    ap.add_argument('--cutoff-s', type=float, default=1.0)
    args = ap.parse_args()

    problem = pickle.loads(args.problem.read_bytes())
    backend = BundledHighsCAPIBackend(
        solver=args.solver,
        threads=1,
        time_limit_s=args.cutoff_s,
        presolve=True,
        enable_basis_hot_start=False,
    )
    try:
        t0 = perf_counter()
        result = execute(problem, backend)
        execute_wall_s = perf_counter() - t0
        terminal = result.status in TERMINAL
        if result.x is not None:
            valid = bool(result.validation is not None and result.validation.valid)
        else:
            valid = bool(terminal)
        raw = dict(result.raw_statistics or {})
        out = {
            'execute_wall_s': float(execute_wall_s),
            'status': result.status.value,
            'objective': None if result.objective is None else float(result.objective),
            'terminal': bool(terminal),
            'valid': bool(valid),
            'solver': args.solver,
            'raw_statistics': raw,
        }
        print(json.dumps(out, sort_keys=True))
    finally:
        backend.close()
        backend.reset_global_scheduler(blocking=True)


if __name__ == '__main__':
    main()
