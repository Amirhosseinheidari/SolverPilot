"""Replay consumed public cases; this is certificate diagnosis, not held-out routing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from solverpilot import read_mps, solve
from solverpilot.backends import HighspyNativeBackend
from solverpilot.validate.lp_dual import recover_lp_optimality
from prepare_public_lp import relax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--corpus', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise SystemExit('refusing to overwrite evidence')
    records = []
    for name in ('netlib-pilot4', 'miplib-unitcal_7'):
        problem = relax(read_mps(args.corpus / (name + '.mps')))
        result = solve(problem, backend=HighspyNativeBackend(solver='simplex', threads=1, time_limit_s=5))
        dual = result.raw_statistics.get('canonical_dual')
        start = perf_counter()
        check = recover_lp_optimality(problem, result.x, dual) if dual is not None and result.x is not None else None
        records.append(dict(instance=name, data_hash=problem.data_hash,
            default_verified=result.optimality_evidence.independently_verified_optimal,
            extended_check=None if check is None else asdict(check), recovery_s=perf_counter()-start))
    # Nonfinite diagnostic bounds are strings, never valid JSON numeric claims.
    def clean(value):
        import math
        if isinstance(value, float) and not math.isfinite(value): return str(value)
        if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
        return value
    payload = {'scope': __doc__, 'observations': [clean(row) for row in records]}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps(payload, allow_nan=False))


if __name__ == '__main__': main()
