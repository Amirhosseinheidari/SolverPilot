"""Frozen public training and fresh held-out qualification, without test tuning.

Consumed stage4 cases are development/training only. New cases must be disjoint
by identity, declared family and canonical hash. All failures count as PAR10.
"""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys

from solverpilot.benchmark.environment import capture_environment, benchmark_environment_fingerprint, thread_environment
from solverpilot.experimental.learned_lp import LPObservation, fit_lp_selector
from public_lp_worker import CANDIDATES, PACKAGES
from qualify_public_lp import run_process, sha, write, summarize, REPEATS, CUTOFF, DEADLINE


STRATEGIES = ('robust', 'default', 'production', *CANDIDATES)


def verify_split(training, heldout):
    for key in ('name', 'group', 'data_hash'):
        if {c[key] for c in training} & {c[key] for c in heldout}:
            raise ValueError('training/held-out overlap: '+key)
    if len(training) != 24 or len(heldout) != 24:
        raise ValueError('24 public training and 24 held-out instances required')
    for cases in (training, heldout):
        for key in ('name', 'group', 'data_hash'):
            if len({c[key] for c in cases}) != len(cases):
                raise ValueError('duplicate cohort '+key)


def training_rows(observations, cases, environment_id):
    expected = {(c['file'], s, r) for c in cases for s in CANDIDATES for r in range(REPEATS)}
    actual = [(r['instance'], r['strategy'], r['repeat']) for r in observations]
    if len(actual) != len(set(actual)) or set(actual) != expected:
        raise ValueError('incomplete or duplicate training outcomes')
    result = []
    for c in cases:
        rows = [r for r in observations if r['instance'] == c['file']]
        features = [r['features'] for r in rows if 'features' in r]
        if not features or any(f != features[0] for f in features):
            raise ValueError('missing/inconsistent outcome-free features')
        verified = [r['objective'] for r in rows if r.get('verified')]
        ref = c.get('reference')
        if verified and any(abs(v-verified[0]) > 1e-6*max(1, abs(verified[0])) for v in verified):
            raise ValueError('training verified objective mismatch')
        if ref is not None and any(abs(v-ref) > 1e-6*max(1, abs(ref)) for v in verified):
            raise ValueError('training reference mismatch')
        samples = {s: tuple((r['wall_s'], bool(r.get('verified') and
                    r.get('api_wall_s', float('inf')) <= CUTOFF and r['wall_s'] <= DEADLINE))
                    for r in rows if r['strategy'] == s) for s in CANDIDATES}
        result.append(LPObservation(c['file'], c['group'], c['data_hash'], 'train',
                                  environment_id, tuple(features[0]), samples))
    return result


def public_summary(rows, cases):
    # Reuse the preregistered stage4 complete-accounting gate, with the new policy.
    report = summarize([{**r, 'strategy': 'learned' if r['strategy'] == 'robust' else r['strategy']}
                        for r in rows], cases)
    for key in ('mean_par10_s', 'verified_within_budget', 'independently_verified_total'):
        report[key]['robust'] = report[key].pop('learned')
    report['policy'] = 'public-trained stump with production fallback on abstention'
    return report


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--training', type=Path, required=True)
    ap.add_argument('--heldout', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()
    if os.name != 'posix' or not hasattr(os, 'sched_setaffinity'):
        raise SystemExit('Linux/WSL process groups and affinity required')
    train = json.loads((args.training/'cohort.json').read_text())
    test = json.loads((args.heldout/'cohort.json').read_text())
    verify_split(train['selected'], test['selected'])
    for folder, cohort in ((args.training, train), (args.heldout, test)):
        for c in cohort['selected']:
            if sha(folder/c['file']) != c['mps_sha256']:
                raise ValueError('corpus bytes changed')
    args.output.mkdir(parents=True, exist_ok=False)
    cpu = min(os.sched_getaffinity(0)); os.sched_setaffinity(0, {cpu})
    env = dict(os.environ); env.update(thread_environment(1))
    environment = capture_environment(packages=PACKAGES)
    if any(environment['env_threads'].get(k) != v for k, v in thread_environment(1).items()):
        raise ValueError('set thread environment to one before starting the parent')
    env_id = benchmark_environment_fingerprint(environment, thread_env_limit=1,
                                               solver_threads=1, worker_python_mode='normal')
    root = Path(__file__).resolve().parents[1]
    worker = Path(__file__).with_name('robust_lp_worker.py')
    sources = ['benchmarks/qualify_robust_lp.py', 'benchmarks/robust_lp_worker.py',
        'benchmarks/qualify_public_lp.py', 'benchmarks/public_lp_worker.py',
        'benchmarks/prepare_public_lp.py', 'src/solverpilot/experimental/learned_lp.py',
        'src/solverpilot/experimental/robust_lp.py', 'src/solverpilot/experimental/lp_environment.py',
        'src/solverpilot/validate/lp_dual.py', 'src/solverpilot/validate/optimality.py',
        'src/solverpilot/runtime/executor.py']
    protocol = {'schema': 'solverpilot.robust-public-lp.v1', 'captured_at': datetime.now(timezone.utc).isoformat(),
        'cpu_only': True, 'environment': environment, 'environment_id': env_id, 'affinity_cpu': cpu,
        'training_cohort_sha256': sha(args.training/'cohort.json'), 'heldout_cohort_sha256': sha(args.heldout/'cohort.json'),
        'source_hashes': {p: sha(root/p) for p in sources}, 'api_budget_s': CUTOFF,
        'controller_deadline_s': DEADLINE, 'repeats': REPEATS, 'parallel_solves': 1,
        'training': '24 consumed public stage4 models; no synthetic-only baseline; min_leaf_groups=4; margin=.005',
        'cost': 'all repeats: full process wall if independently verified within 2s API and 12s process; otherwise 120s',
        'training_cutoff_s': DEADLINE, 'abstention': 'ordinary solve_production with remaining budget',
        'gate': 'stage4 preregistered gate against both default APIs; >=20 groups; actual learned switches; no verified objective mismatch; ratio<=.97; bootstrap upper<1; p90<=1.25; no lost successes',
        'retuning_allowed': False, 'automatic_production_routing_enabled': False}
    write(args.output/'protocol.json', protocol)
    write(args.output/'training-cohort.json', train)
    write(args.output/'heldout-cohort.json', test)
    model_path = args.output/'model.json'

    def collect(folder, cases, strategies, filename):
        rows = []
        for i, c in enumerate(cases):
            for repeat in range(REPEATS):
                shift = i % len(strategies); order = list(strategies[shift:]+strategies[:shift])
                if repeat: order.reverse()
                for s in order:
                    command = [sys.executable, str(worker), '--mps', str(folder/c['file']),
                        '--strategy', s, '--model', str(model_path), '--cutoff', str(CUTOFF)]
                    r = run_process(command, env, DEADLINE)
                    if 'data_hash' in r and r['data_hash'] != c['data_hash']:
                        raise ValueError('worker model identity mismatch')
                    rows.append({'instance': c['file'], 'strategy': s, 'repeat': repeat, **r})
                    write(args.output/filename, rows)
            print(json.dumps({'phase': filename, 'completed': c['file'], 'outcomes': len(rows)}), flush=True)
        return rows

    raw = collect(args.training, train['selected'], CANDIDATES, 'training-observations.json')
    rows = training_rows(raw, train['selected'], env_id)
    write(args.output/'training-rows.json', [asdict(r) for r in rows])
    model = fit_lp_selector(rows, candidates=CANDIDATES, cutoff_s=DEADLINE,
                            protocol_sha256=sha(args.output/'protocol.json'), min_leaf_groups=4, switch_margin_s=.005)
    model.save(model_path)
    write(args.output/'model-freeze.json', {'sha256': sha(model_path), 'before_any_heldout_solve': True,
                                         'captured_at': datetime.now(timezone.utc).isoformat()})
    raw = collect(args.heldout, test['selected'], STRATEGIES, 'heldout-observations.json')
    write(args.output/'summary.json', public_summary(raw, test['selected']))
    stress = []
    for case in ('infeasible', 'unbounded', 'ill_scaled', 'deadline'):
        for s in STRATEGIES:
            cmd = [sys.executable, str(worker), '--stress', case, '--strategy', s,
                   '--model', str(model_path), '--cutoff', str(.0001 if case == 'deadline' else CUTOFF)]
            stress.append({'case': case, 'strategy': s, **run_process(cmd, env, DEADLINE)})
            write(args.output/'stress.json', stress)
    print(json.dumps(json.loads((args.output/'summary.json').read_text()), indent=2), flush=True)


if __name__ == '__main__':
    main()
