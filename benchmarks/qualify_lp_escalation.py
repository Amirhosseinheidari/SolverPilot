"""Explicit native exact fallback replay; not fresh routing qualification."""
import argparse
from dataclasses import asdict
import json
import math
from pathlib import Path
from fractions import Fraction

from solverpilot import read_mps, solve
from solverpilot.backends import HighspyNativeBackend
from solverpilot.validate.escalation import recover_lp_with_fallback
from prepare_public_lp import relax


def clean(value):
    if isinstance(value, Fraction): return str(value)
    if isinstance(value, float) and not math.isfinite(value): return str(value)
    if isinstance(value, dict): return {k: clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)): return [clean(v) for v in value]
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mps', type=Path, required=True)
    parser.add_argument('--solver', required=True)
    parser.add_argument('--checker', required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--evidence', type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists(): raise ValueError('refusing to overwrite evidence')
    p = relax(read_mps(args.mps))
    r = solve(p, backend=HighspyNativeBackend(solver='simplex', threads=1, time_limit_s=5))
    result = recover_lp_with_fallback(p, r.x, r.raw_statistics['canonical_dual'],
        basis=r.raw_statistics.get('lp_basis'), time_limit_s=60., numerical_limit_s=1e-9,
        scip_executable=args.solver, checker_executable=args.checker, evidence_directory=args.evidence)
    payload = {'scope': __doc__, 'data_hash': p.data_hash, 'numerical_phase_deliberately_exhausted': True,
               'result': clean(asdict(result))}
    args.output.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding='utf-8')
    print(json.dumps({'verified': result.optimality_verified, 'kind': result.proof_kind,
        'elapsed_s': result.elapsed_s, 'reason': None if result.exact_result is None else result.exact_result.reason}))


if __name__ == '__main__': main()
