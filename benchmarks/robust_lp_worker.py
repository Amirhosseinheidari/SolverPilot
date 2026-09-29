"""Isolated call for the public-trained policy; no training or retry in workers."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

from solverpilot import read_mps
from solverpilot.benchmark.environment import capture_environment
from solverpilot.experimental.learned_lp import LPSelector, lp_features
from solverpilot.experimental.lp_environment import bind_lp_environment
from solverpilot.experimental.robust_lp import solve_robust_lp
from prepare_public_lp import relax
from public_lp_worker import CANDIDATES, PACKAGES, candidate, run_strategy, stress_problem


def run(p, strategy, model_path, cutoff):
    if strategy != 'robust':
        return run_strategy(p, strategy, model_path, cutoff)
    start = perf_counter()
    model = LPSelector.load(model_path)
    binding = bind_lp_environment(model, Path(model_path).with_name('protocol.json'),
                                  capture_environment(packages=PACKAGES))
    remaining = cutoff-(perf_counter()-start)
    if remaining <= 0:
        return {'status': 'setup_timeout', 'verified': False, 'api_wall_s': perf_counter()-start}
    try:
        result = solve_robust_lp(p, model, environment_id=binding['decision_environment_id'],
            backends={name: candidate(name, remaining) for name in CANDIDATES}, time_limit_s=remaining)
    except TimeoutError:
        return {'status': 'setup_timeout', 'verified': False, 'api_wall_s': perf_counter()-start}
    route = dict(result.raw_statistics['experimental_lp_route'])
    return {'status': result.status.value, 'backend_status': result.backend_status,
            'backend': result.trace.backend, 'objective': result.objective,
            'primal_valid': bool(result.validation and result.validation.valid),
            'verified': result.optimality_evidence.independently_verified_optimal,
            'api_wall_s': perf_counter()-start, 'trace_timings': asdict(result.trace.timings),
            'decision': {'reason': route['learned_reason'], 'candidate': route['candidate']},
            'route': route, 'environment_binding': binding}


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--mps', type=Path)
    ap.add_argument('--stress', choices=('infeasible', 'unbounded', 'ill_scaled', 'deadline'))
    ap.add_argument('--model', type=Path, required=True)
    ap.add_argument('--strategy', choices=('robust', 'default', 'production', *CANDIDATES), required=True)
    ap.add_argument('--cutoff', type=float, default=2.)
    args = ap.parse_args()
    p = stress_problem(args.stress) if args.stress else relax(read_mps(args.mps))
    result = run(p, args.strategy, args.model, args.cutoff)
    print(json.dumps({**result, 'data_hash': p.data_hash, 'features': lp_features(p)}, allow_nan=False), flush=True)


if __name__ == '__main__':
    main()
