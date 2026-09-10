from __future__ import annotations

import argparse
import hashlib
import json
import os
import pickle
import statistics
import subprocess
import sys
from pathlib import Path
from time import perf_counter

from solverpilot.benchmark.environment import (
    benchmark_environment_fingerprint,
    capture_environment,
    thread_environment,
)
from solverpilot.benchmark.runner import _clean_pythonpath

ROUNDS = 3
CUTOFF_S = 1.0
HARD_TIMEOUT_S = 3.0
SOLVERS = ('choose', 'simplex', 'ipm')


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def sample(worker: Path, cache: Path, solver: str, root: Path, env: dict[str, str]) -> dict:
    try:
        proc = subprocess.run(
            [sys.executable, '-S', str(worker), '--problem', str(cache), '--solver', solver, '--cutoff-s', str(CUTOFF_S)],
            cwd=root,
            env=env,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            timeout=HARD_TIMEOUT_S,
            check=False,
        )
    except subprocess.TimeoutExpired:
        return {
            'execute_wall_s': HARD_TIMEOUT_S,
            'status': 'hard_timeout',
            'objective': None,
            'terminal': False,
            'valid': False,
            'controller_hard_timeout': True,
        }
    if proc.returncode != 0:
        return {
            'execute_wall_s': HARD_TIMEOUT_S,
            'status': 'worker_error',
            'objective': None,
            'terminal': False,
            'valid': False,
            'stderr': proc.stderr[-2000:],
        }
    try:
        return json.loads(proc.stdout.strip().splitlines()[-1])
    except Exception as exc:
        return {
            'execute_wall_s': HARD_TIMEOUT_S,
            'status': 'worker_parse_error',
            'objective': None,
            'terminal': False,
            'valid': False,
            'stderr': f'{type(exc).__name__}: {exc}; stdout={proc.stdout[-1000:]}; stderr={proc.stderr[-1000:]}',
        }


def cost(sample_row: dict) -> float:
    return min(float(sample_row['execute_wall_s']), CUTOFF_S) if sample_row['terminal'] else CUTOFF_S


def algorithm_used(sample_row: dict) -> str:
    raw = sample_row.get('raw_statistics') or {}
    simp = int(raw.get('simplex_iteration_count') or 0)
    ipm = int(raw.get('ipm_iteration_count') or 0)
    if ipm > 0 and simp == 0:
        return 'ipm'
    if simp > 0 and ipm == 0:
        return 'simplex'
    if simp > 0 and ipm > 0:
        return 'mixed_or_crossover'
    return 'presolve_or_undetermined'


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--cohort', type=Path, required=True)
    ap.add_argument('--shard-index', type=int, required=True)
    ap.add_argument('--shard-count', type=int, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--m22-integrity', type=Path, required=True)
    ap.add_argument('--protocol', type=Path, required=True)
    args = ap.parse_args()

    cohort = json.loads(args.cohort.read_text(encoding='utf-8'))
    full = cohort['selected']
    chosen = [row for i, row in enumerate(full) if i % args.shard_count == args.shard_index]
    root = Path(__file__).resolve().parents[1]
    worker = root / 'benchmarks' / 'm28_native_choose_worker.py'
    runner = Path(__file__).resolve()

    integ = json.loads(args.m22_integrity.read_text(encoding='utf-8'))
    mip = (integ.get('archives') or {}).get('miplib') or {}
    integrity_ok = bool(integ.get('all_integrity_pass')) and mip.get('sha256_match') is True and mip.get('sha256') == cohort['miplib_zip_sha256']

    env = dict(os.environ)
    env.update(thread_environment(1))
    env['PYTHONPATH'] = _clean_pythonpath()
    captured = capture_environment(packages=('numpy', 'scipy', 'casadi'))
    timing_fp = benchmark_environment_fingerprint(captured, thread_env_limit=1, solver_threads=1, worker_python_mode='no_site')
    fixed_host_payload = {
        'hostname': captured.get('hostname'),
        'timing_environment_fingerprint': timing_fp,
        'effective_thread_env': thread_environment(1),
        'solver_threads': 1,
        'worker_python_mode': 'no_site',
    }
    fixed_host_fp = hashlib.sha256(json.dumps(fixed_host_payload, sort_keys=True, separators=(',', ':')).encode()).hexdigest()

    file_hashes = {
        'cohort_sha256': sha256_file(args.cohort),
        'protocol_sha256': sha256_file(args.protocol),
        'runner_sha256': sha256_file(runner),
        'worker_sha256': sha256_file(worker),
        'm22_integrity_sha256': sha256_file(args.m22_integrity),
    }

    rows: list[dict] = []
    started = perf_counter()
    for item in chosen:
        global_idx = next(i for i, x in enumerate(full) if x['instance'] == item['instance'])
        samples = {s: [] for s in SOLVERS}
        for rd in range(ROUNDS):
            order = list(SOLVERS)
            shift = (global_idx + rd) % len(order)
            order = order[shift:] + order[:shift]
            for solver in order:
                samples[solver].append(sample(worker, Path(item['cache']), solver, root, env))

        medians = {s: float(statistics.median(cost(x) for x in samples[s])) for s in SOLVERS}
        objective_medians: dict[str, float | None] = {}
        for solver in SOLVERS:
            vals = [x['objective'] for x in samples[solver] if x.get('status') == 'valid_optimal' and x.get('objective') is not None]
            objective_medians[solver] = None if not vals else float(statistics.median(vals))
        objective_spread = None
        present = [v for v in objective_medians.values() if v is not None]
        if len(present) >= 2:
            objective_spread = float(max(present) - min(present))

        choose_algorithms = [algorithm_used(x) for x in samples['choose']]
        rows.append({
            'instance': item['instance'],
            'cache_sha256': item['cache_sha256'],
            'n': item['n'],
            'm': item['m'],
            'nnz': item['nnz'],
            'cost_medians_s': medians,
            'objective_medians': objective_medians,
            'objective_spread': objective_spread,
            'choose_algorithm_diagnostics': choose_algorithms,
            'samples': samples,
        })

    payload = {
        'schema': 'optimind.m28.native_choose_shard.v1',
        'source': cohort['source'],
        'cohort_signature': cohort['cohort_signature'],
        'full_cohort_names': [x['instance'] for x in full],
        'fresh_overlap_count': cohort['fresh_overlap_count'],
        'miplib_zip_sha256': cohort['miplib_zip_sha256'],
        'manifest_sha256': cohort['manifest_sha256'],
        'prior_cohort_file_sha256': cohort['prior_cohort_file_sha256'],
        'official_corpus_integrity_passed': integrity_ok,
        'rounds': ROUNDS,
        'cutoff_s': CUTOFF_S,
        'hard_timeout_s': HARD_TIMEOUT_S,
        'solver_threads': 1,
        'thread_env_limit': 1,
        'algorithms': list(SOLVERS),
        'shard_index': args.shard_index,
        'shard_count': args.shard_count,
        'file_hashes': file_hashes,
        'environment': captured,
        'timing_environment_fingerprint': timing_fp,
        'fixed_host_fingerprint': fixed_host_fp,
        'rows': rows,
        'wall_s': perf_counter() - started,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    counts = {s: {} for s in SOLVERS}
    for row in rows:
        for s in SOLVERS:
            for x in row['samples'][s]:
                counts[s][x['status']] = counts[s].get(x['status'], 0) + 1
    print(json.dumps({'shard': args.shard_index, 'rows': len(rows), 'wall_s': payload['wall_s'], 'status_counts': counts}, indent=2))


if __name__ == '__main__':
    main()
