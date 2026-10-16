"""Replay every frozen public candidate/dual without asking a solver for new ones.

This is a new verifier protocol over the existing admission, not a new corpus.
Original tolerance checks and all failures remain visible. The source/wheel used
by the caller must be recorded alongside the resulting JSON.
"""
import argparse
from collections import Counter
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from qualify_public_reuse import (SCHEMA, STEPS, TOL, check_raw, row_key, sha,
                                  transformed_problem, write_json)
from solverpilot import read_mps
from solverpilot._identity import process_source_sha256
from solverpilot.benchmark.environment import capture_environment
from solverpilot.validate.lp_dual import prepare_lp_dual
from solverpilot.validate.optimality import verify_optimality


def independently_verified(evidence):
    return bool(evidence.get('primal_validated') is True and evidence.get('dual_verified') is True and
                (evidence.get('gap_verified') is True or evidence.get('certificate_verified') is True))


def replay_one(old, case, raw, problem):
    row = {'key': list(row_key(old)), 'problem_data_hash': problem.data_hash,
           'previously_verified': False, 'original_candidate_check': {'passed': False},
           'verified': False}
    started = perf_counter()
    try:
        row['previously_verified'] = independently_verified(old.get('evidence', {}))
        x = np.array(old['candidate'], dtype=float)
        dual = np.array(old['statistics']['canonical_dual'], dtype=float)
        original_check = check_raw(raw, old['step'], x, old['objective'], case['reference'])
        row['original_candidate_check'] = original_check
        checked = verify_optimality(problem, x, dual, fast_reject=True)
        row['raw_check'] = asdict(checked)
        if checked.reason == 'dual multiplier points toward an infinite bound':
            prepared = prepare_lp_dual(problem, dual)
            if prepared is not None:
                checked = verify_optimality(problem, x, prepared, fast_reject=True)
                row['prepared_check'] = asdict(checked)
        row['verified'] = bool(original_check['passed'] and checked.verified)
        row['check'] = asdict(checked)
    except Exception as exc:  # Preserve missing/malformed historical candidates too.
        row['error'] = f'{type(exc).__name__}: {exc}'
    # Includes original-data checking, dual preparation/retry and serialization;
    # model construction is outside. Compare only through this identical runner.
    row['replay_s'] = perf_counter()-started
    return row


def load_inputs(prepared, previous):
    freeze = json.loads((prepared/'freeze.json').read_text())
    if freeze['schema'] != SCHEMA or set(freeze['files']) != {'protocol.json', 'cohort.json'}:
        raise ValueError('unsupported frozen admission')
    for name, digest in freeze['files'].items():
        if sha((prepared/name).read_bytes()) != digest:
            raise ValueError('frozen admission bytes changed')
    protocol = json.loads((prepared/'protocol.json').read_text())
    cohort = json.loads((prepared/'cohort.json').read_text())
    if protocol['steps'] != list(STEPS) or protocol['tolerances'] != TOL or not cohort['complete']:
        raise ValueError('unsupported or incomplete frozen admission')
    cases = {}
    for case in cohort['selected']:
        if Path(case['file']).name != case['file']:
            raise ValueError('invalid model filename')
        path = prepared/case['file']
        if sha(path.read_bytes()) != case['mps_sha256']:
            raise ValueError('frozen model bytes changed')
        raw = read_mps(path)
        if raw.data_hash != case['data_hash']:
            raise ValueError('frozen model identity changed')
        cases[case['name']] = (case, raw)
    old = json.loads(previous.read_text())
    expected = {(c['name'], r, p, mode, step) for c in cohort['selected']
                for r in range(protocol['repeats']) for p in protocol['presolve']
                for mode in protocol['modes'] for step in protocol['steps']}
    keys = [row_key(row) for row in old['rows']]
    if set(keys) != expected or len(keys) != len(expected):
        raise ValueError('missing, extra or duplicate historical observations')
    return protocol, cases, old['rows']


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--prepared', type=Path, required=True)
    parser.add_argument('--previous', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--require-all', action='store_true')
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    protocol, cases, historical = load_inputs(args.prepared, args.previous)
    models = {(name, step): transformed_problem(raw, step)
              for name, (_, raw) in cases.items() for step in STEPS}
    rows = []
    for old in historical:
        case, raw = cases[old['name']]
        problem = models[old['name'], old['step']]
        rows.append(replay_one(old, case, raw, problem))
    losses = sum(r['previously_verified'] and not r['verified'] for r in rows)
    gained = sum(not r['previously_verified'] and r['verified'] for r in rows)
    counts = {name: dict(Counter('verified' if r['verified'] else 'unverified'
                                for r in rows if r['key'][0] == name)) for name in cases}
    passed = all(r['original_candidate_check']['passed'] and 'error' not in r for r in rows)
    passed = passed and not losses and (not args.require_all or all(r['verified'] for r in rows))
    summary = {'passed': passed, 'observed': len(rows), 'verified': sum(r['verified'] for r in rows),
               'previously_verified': sum(r['previously_verified'] for r in rows),
               'verification_gains': gained, 'verification_losses': losses, 'families': counts}
    write_json(args.output, {'schema': 'solverpilot.fixed-certificate-replay.v1',
        'summary': summary, 'source_sha256': process_source_sha256(),
        'environment': capture_environment(packages=('numpy', 'scipy', 'highspy')),
        'runner_sha256': sha(Path(__file__).read_bytes()),
        'helper_sha256': sha(Path(__file__).with_name('qualify_public_reuse.py').read_bytes()),
        'previous_sha256': sha(args.previous.read_bytes()),
        'admission_sha256': sha((args.prepared/'freeze.json').read_bytes()),
        'historical_protocol': protocol, 'rows': rows})
    print(json.dumps(summary))
    return 0 if passed else 2


if __name__ == '__main__':
    raise SystemExit(main())
