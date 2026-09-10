from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path

import numpy as np

TARGET = 48
ROUNDS = 3
SOLVERS = ('choose', 'simplex', 'ipm')


def paired_bootstrap(a: np.ndarray, b: np.ndarray, *, draws: int = 30000) -> dict:
    rng = np.random.default_rng(280028)
    n = len(a)
    vals = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(a[idx].mean() / max(b[idx].mean(), 1e-15))
    q = np.quantile(vals, [0.025, 0.5, 0.975])
    return {'point': float(a.mean()/max(b.mean(),1e-15)), 'low': float(q[0]), 'median': float(q[1]), 'high': float(q[2]), 'draws': draws}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--inputs', type=Path, nargs='+', required=True)
    ap.add_argument('--output', type=Path, required=True)
    args = ap.parse_args()

    parts = [json.loads(p.read_text(encoding='utf-8')) for p in args.inputs]
    sigs = {p['cohort_signature'] for p in parts}
    counts = {p['shard_count'] for p in parts}
    idxs = sorted(p['shard_index'] for p in parts)
    host_fps = {p['fixed_host_fingerprint'] for p in parts}
    timing_fps = {p['timing_environment_fingerprint'] for p in parts}
    file_hash_sets = {json.dumps(p['file_hashes'], sort_keys=True) for p in parts}
    if len(sigs) != 1 or len(counts) != 1:
        raise SystemExit('cohort/shard mismatch')
    shard_count = next(iter(counts))
    if idxs != list(range(shard_count)):
        raise SystemExit(f'incomplete shards: {idxs}')
    if len(host_fps) != 1 or len(timing_fps) != 1:
        raise SystemExit('mixed host/environment evidence')
    if len(file_hash_sets) != 1:
        raise SystemExit('runner/protocol/cohort hashes differ across shards')

    rows = [r for p in parts for r in p['rows']]
    names = [r['instance'] for r in rows]
    if len(rows) != TARGET or len(set(names)) != TARGET:
        raise SystemExit(f'expected {TARGET} unique rows, got {len(rows)}/{len(set(names))}')
    if set(names) != set(parts[0]['full_cohort_names']):
        raise SystemExit('merged rows do not equal frozen cohort')

    cost = {s: np.asarray([r['cost_medians_s'][s] for r in rows], dtype=float) for s in SOLVERS}
    means = {s: float(cost[s].mean()) for s in SOLVERS}
    forced_means = {'simplex': means['simplex'], 'ipm': means['ipm']}
    sbs = min(forced_means, key=forced_means.get)
    sbs_cost = cost[sbs]
    vbs = np.minimum(cost['simplex'], cost['ipm'])
    gap = float((sbs_cost.mean() - vbs.mean()) / max(sbs_cost.mean(), 1e-15))
    choose_to_sbs = paired_bootstrap(cost['choose'], sbs_cost)
    choose_gain_over_sbs = float((sbs_cost.mean()-cost['choose'].mean())/max(sbs_cost.mean(),1e-15))
    choose_gap_closure = None
    if gap > 0:
        choose_gap_closure = float((sbs_cost.mean()-cost['choose'].mean())/max(sbs_cost.mean()-vbs.mean(),1e-15))

    total = TARGET * ROUNDS
    terminal_rates = {}
    invalid_counts = {}
    status_counts = {}
    for s in SOLVERS:
        samples = [x for r in rows for x in r['samples'][s]]
        terminal_rates[s] = sum(bool(x['terminal']) for x in samples) / total
        invalid_counts[s] = sum((x.get('terminal') and not x.get('valid')) or (x.get('objective') is not None and not x.get('valid')) for x in samples)
        sc = {}
        for x in samples:
            sc[x['status']] = sc.get(x['status'], 0) + 1
        status_counts[s] = sc

    objective_mismatch = 0
    for r in rows:
        vals = [v for v in r['objective_medians'].values() if v is not None]
        if len(vals) < 2:
            continue
        ref = vals[0]
        tol = 1e-6 * max(1.0, abs(ref))
        if max(vals)-min(vals) > tol:
            objective_mismatch += 1

    choose_diag = {}
    nontrivial_choose = {'simplex': 0, 'ipm': 0, 'mixed_or_crossover': 0, 'presolve_or_undetermined': 0}
    for r in rows:
        for label in r['choose_algorithm_diagnostics']:
            choose_diag[label] = choose_diag.get(label, 0) + 1
            nontrivial_choose[label] = nontrivial_choose.get(label, 0) + 1
    both_algorithms_observed = nontrivial_choose.get('simplex',0) > 0 and nontrivial_choose.get('ipm',0) > 0

    positive_gap = gap >= 0.01
    gates = {
        'official_corpus_integrity': all(p['official_corpus_integrity_passed'] for p in parts),
        'fresh_48_unique_zero_prior_overlap': len(rows) == 48 and parts[0]['fresh_overlap_count'] == 0,
        'complete_three_round_accounting': all(sum(len(r['samples'][s]) for r in rows) == total for s in SOLVERS),
        'choose_terminal_rate_ge_0_90': terminal_rates['choose'] >= 0.90,
        'simplex_terminal_rate_ge_0_90': terminal_rates['simplex'] >= 0.90,
        'ipm_terminal_rate_ge_0_90': terminal_rates['ipm'] >= 0.90,
        'zero_independent_invalidity': sum(invalid_counts.values()) == 0,
        'zero_material_objective_mismatch': objective_mismatch == 0,
        'choose_mean_to_sbs_le_1_02': choose_to_sbs['point'] <= 1.02,
        'choose_bootstrap_upper_lt_1_05': choose_to_sbs['high'] < 1.05,
        'choose_closes_ge_80pct_gap_when_gap_ge_1pct': (not positive_gap) or (choose_gap_closure is not None and choose_gap_closure >= 0.80),
        'single_fixed_host_and_runner_provenance': len(host_fps) == 1 and len(file_hash_sets) == 1,
    }
    closes_opportunity = all(gates.values())

    payload = {
        'schema': 'optimind.m28.native_choose_baseline_audit.v1',
        'purpose': 'fresh public/OOD audit of native HiGHS choose vs forced simplex/IPM before any further learned selector work',
        'source': parts[0]['source'],
        'instances': len(rows),
        'rounds': ROUNDS,
        'cohort_signature': next(iter(sigs)),
        'miplib_zip_sha256': parts[0]['miplib_zip_sha256'],
        'manifest_sha256': parts[0]['manifest_sha256'],
        'prior_cohort_file_sha256': parts[0]['prior_cohort_file_sha256'],
        'file_hashes': parts[0]['file_hashes'],
        'environment': parts[0]['environment'],
        'timing_environment_fingerprint': parts[0]['timing_environment_fingerprint'],
        'fixed_host_fingerprint': parts[0]['fixed_host_fingerprint'],
        'metrics': {
            'mean_cost_s': means,
            'forced_sbs_backend': sbs,
            'forced_sbs_mean_s': float(sbs_cost.mean()),
            'forced_vbs_mean_s': float(vbs.mean()),
            'forced_sbs_to_vbs_relative_gap': gap,
            'choose_relative_gain_vs_sbs': choose_gain_over_sbs,
            'choose_to_sbs_bootstrap': choose_to_sbs,
            'choose_vbs_gap_closure': choose_gap_closure,
            'terminal_rates': terminal_rates,
            'invalid_counts': invalid_counts,
            'objective_mismatch_instances': objective_mismatch,
            'status_counts': status_counts,
            'choose_algorithm_diagnostics': choose_diag,
            'both_simplex_and_ipm_observed_under_choose': both_algorithms_observed,
        },
        'pre_registered_gates': gates,
        'native_choose_closes_portfolio_opportunity': closes_opportunity,
        'claims_boundary': [
            'Timing comparison is within bundled HiGHS 1.10.0 only; it is not a benchmark of current highspy 1.15.x.',
            'SciPy method=highs is handled separately as a semantics probe and is not an independent portfolio competitor.',
            'If choose uses only simplex on nontrivial cases, M28 must not call it an instance-wise algorithm selector.',
            'M28 does not train or promote an OptiMind learned selector.',
        ],
        'rows': sorted(rows, key=lambda r: r['instance']),
        'shards': [{'index': p['shard_index'], 'rows': len(p['rows']), 'wall_s': p['wall_s']} for p in parts],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({'metrics': payload['metrics'], 'gates': gates, 'closes_opportunity': closes_opportunity}, indent=2))


if __name__ == '__main__':
    main()
