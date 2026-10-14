"""Frozen gates for actual public LP routing calls, never activation authority.

Shadow predictions and a fast production fallback do not demonstrate learned
routing value. This evaluator requires complete, paired wrapper observations.
"""
from collections import Counter
import math
from pathlib import Path
from urllib.parse import urlparse

import numpy as np

PUBLIC_SCHEMA = 'solverpilot.lp-v2-public-manifest.v1'
POLICY = dict(min_training_families=4, min_calibration_families=4,
              min_test_families=20, min_switched_families=4,
              max_mean_ratio=.97, max_p90_ratio=1.25,
              bootstrap_draws=2000, bootstrap_seed=271828)


def check_loaded_source(package, expected):
    """Fail rather than label an installed/stale package as this checkout."""
    import sys
    expected = Path(expected).resolve()
    roots = [Path(p).resolve() for p in package.__path__]
    if roots != [expected]:
        raise ValueError('executed solverpilot source differs from frozen checkout')
    for name, module in tuple(sys.modules.items()):
        if name == 'solverpilot' or name.startswith('solverpilot.'):
            location = getattr(module, '__file__', None)
            if location is not None and not Path(location).resolve().is_relative_to(expected):
                raise ValueError('mixed solverpilot source origins')


def public_preflight(manifest, cases, config):
    """Require novelty declarations and adequate families before any solve."""
    from public_lp_families import conservative_family as family
    if not manifest.get('consumed_lists'):
        raise ValueError('public qualification requires frozen consumed identity lists')
    if config['repeats'] % 2:
        raise ValueError('public repeat count must be even for paired order')
    for split, minimum in (('train', POLICY['min_training_families']),
                           ('calibration', POLICY['min_calibration_families']),
                           ('test', POLICY['min_test_families'])):
        count = len({c['family'] for c in cases[split]})
        if count < minimum:
            raise ValueError(f'public {split} needs {minimum} independent families; found {count}')
        for case in cases[split]:
            if case.get('format') != 'mps':
                raise ValueError('public input must be a materialized continuous MPS file')
            if case.get('reference') is None:
                raise ValueError('public case needs an independent objective reference')
            if not isinstance(case.get('source_url'), str) or not case['source_url'].startswith('https://'):
                raise ValueError('public case needs an HTTPS source URL')
            if case.get('source') not in {'netlib', 'external'}:
                raise ValueError('public case source must be netlib or external')
            if (urlparse(case['source_url']).hostname in {'netlib.org', 'www.netlib.org'}
                    and case['source'] != 'netlib'):
                raise ValueError('Netlib source must use canonical Netlib family checks')
            if not isinstance(case.get('reference_url'), str) or not case['reference_url'].startswith('https://'):
                raise ValueError('public case needs an HTTPS reference URL')
            if case.get('source') == 'netlib':
                name = case['name'].removeprefix('netlib-')
                if case['family'] != 'netlib:'+family(name):
                    raise ValueError('public Netlib case has a noncanonical family alias')


def evaluate_actual_routes(rows, cases, config, *, baseline, model_sha256, guard_sha256):
    """All reported costs are observed wrapper calls plus declared setup shares."""
    from qualify_lp_v2 import score_outcome, validate_outcome_coverage
    validate_outcome_coverage(rows, cases, config['repeats'], 'test')
    by_case = {c['name']: c for c in cases}
    if len(by_case) != len(cases):
        raise ValueError('duplicate admitted test identity')
    for row in rows:
        case = by_case[row['name']]
        if (row['family'] != case['family'] or row['data_hash'] != case['data_hash']
                or row['split'] != 'test'):
            raise ValueError('outcome identity differs from admitted test case')
    if baseline not in ('highs-simplex', 'highs-ipm'):
        raise ValueError('training-frozen single solver must be a measured candidate')
    names = sorted(by_case)
    families = sorted({c['family'] for c in cases})
    indices = [[i for i, name in enumerate(names) if by_case[name]['family'] == f] for f in families]
    lookup = {(r['name'], r['strategy'], r['repeat']): r for r in rows}
    routed = [r for r in rows if r['strategy'] == 'v2']
    binding_ok = all(r.get('route', {}).get('model_sha256') == model_sha256
                     and r.get('route', {}).get('gain_guard_sha256') == guard_sha256
                     for r in routed)
    switches = [r for r in routed if r.get('route', {}).get('candidate') in
                ('highs-simplex', 'highs-ipm') and r['route']['candidate'] != baseline
                and r['route'].get('learned_reason') == 'learned_tree_v2'
                and r.get('route', {}).get('reason') == 'in_support'
                and type(r.get('route', {}).get('leaf_id')) is int]
    objective_ok = True
    for name in names:
        case = by_case[name]
        reference = case.get('reference')
        if isinstance(reference, bool) or not isinstance(reference, (int, float)) or not math.isfinite(reference):
            objective_ok = False
            continue
        verified = [r for r in rows if r['name'] == name and r.get('verified') is True]
        for row in verified:
            value = row.get('objective')
            if (isinstance(value, bool) or not isinstance(value, (int, float))
                    or not math.isfinite(value) or row.get('objective_consistent') is not True
                    or abs(value-reference) > 1e-6*max(1., abs(value), abs(reference))):
                objective_ok = False
    common = dict(complete_outcomes=True, artifact_bindings_match=binding_ok,
                  objective_consistency=objective_ok,
                  enough_test_families=len(families) >= POLICY['min_test_families'])
    rng = np.random.default_rng(POLICY['bootstrap_seed'])
    draws = rng.integers(0, len(families), (POLICY['bootstrap_draws'], len(families)))
    def ratio(a, b):
        return np.where((a == 0) & (b == 0), 1., a/np.maximum(b, 1e-12))
    scopes = {}
    for scope in ('cold', 'amortized'):
        def scores(strategy):
            values = [[score_outcome(lookup[name, strategy, repeat], cutoff_s=config['cutoff_s'],
                       reuse_count=config['reuse_count'], scope=scope)
                       for repeat in range(config['repeats'])] for name in names]
            return (np.array([np.mean([v['par10_s'] for v in vv]) for vv in values]),
                    np.array([[v['success'] for v in vv] for vv in values]))
        cost, successes = scores('v2')
        family_cost = np.array([cost[ii].mean() for ii in indices])
        comparisons = {}
        for strategy in dict.fromkeys(('default', 'production', baseline)):
            base_cost, base_success = scores(strategy)
            family_base = np.array([base_cost[ii].mean() for ii in indices])
            boot = ratio(family_cost[draws].sum(axis=1), family_base[draws].sum(axis=1))
            mean_ratio = float(ratio(family_cost.mean(), family_base.mean()))
            tail = float(np.quantile(ratio(cost, base_cost), .9))
            from solverpilot.experimental._repeat_quality import paired_repeat_losses
            lost = paired_repeat_losses(successes, base_success)
            gates = dict(three_percent_gain=mean_ratio <= POLICY['max_mean_ratio'],
                         bootstrap_upper_below_one=float(np.quantile(boot, .975)) < 1,
                         tail_slowdown_bounded=tail <= POLICY['max_p90_ratio'],
                         no_lost_verified_repeats=lost == 0)
            comparisons[strategy] = dict(family_weighted_ratio=mean_ratio,
                bootstrap_95pct=[float(np.quantile(boot, .025)), float(np.quantile(boot, .975))],
                p90_instance_ratio=tail, lost_verified_repeats=lost, gates=gates)
        successful_switch_families = {r['family'] for r in switches if score_outcome(r,
            cutoff_s=config['cutoff_s'], reuse_count=config['reuse_count'], scope=scope)['success']}
        gates = {**common, 'actual_learned_switches': len(successful_switch_families) >= POLICY['min_switched_families'],
                 'comparisons_pass': all(all(c['gates'].values()) for c in comparisons.values())}
        scopes[scope] = dict(gates=gates, comparisons=comparisons,
            verified_repeats=int(successes.sum()), successful_switch_families=len(successful_switch_families),
            qualified=all(gates.values()))
    return dict(schema='solverpilot.lp-v2-actual-public-report.v1', policy=dict(POLICY),
        training_frozen_baseline=baseline, family_count=len(families), scopes=scopes,
        actual_learned_switches=len(switches), route_counts=dict(Counter(r.get('route', {}).get('candidate', 'missing') for r in routed)),
        production_authorized=False, automatic_production_routing_enabled=False,
        scope='Measured public wrapper qualification within frozen admission scope; not industrial deployment authority')
