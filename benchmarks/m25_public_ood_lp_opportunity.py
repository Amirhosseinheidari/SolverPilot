from __future__ import annotations

import argparse, gzip, hashlib, json, os, platform, statistics, zipfile
from pathlib import Path
from time import perf_counter

import numpy as np
import scipy

from solverpilot import LinearProblem, VariableDomain, execute, inspect_problem, parse_mps
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.validate import PublicStatus

# Frozen before M25 outcomes are observed.
TARGET_INSTANCES = 48
MAX_COMPRESSED_BYTES = 512 * 1024
MAX_N = 5000
MAX_M = 5000
MAX_NNZ = 100_000
ROUNDS = 2
CUTOFF_S = 1.0
MIN_RELATIVE_VBS_GAIN = 0.03
MIN_RELATIVE_ORACLE_FEATURE_GAIN = 0.02
MIN_NON_SBS_WINS = 3
MIN_DECISIVE_NON_SBS_WINS = 2
DECISIVE_RATIO = 0.90
BOOTSTRAP_DRAWS = 20_000


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open('rb') as f:
        for c in iter(lambda: f.read(1024 * 1024), b''):
            h.update(c)
    return h.hexdigest()


def _filename_key(name: str) -> str:
    return hashlib.sha256(name.encode('utf-8')).hexdigest()


def _relax(p: LinearProblem, name: str) -> LinearProblem:
    return LinearProblem.from_data(
        A=p.A,
        c=p.c,
        variable_lower=p.variable_lower,
        variable_upper=p.variable_upper,
        constraint_lower=p.constraint_lower,
        constraint_upper=p.constraint_upper,
        domains=[VariableDomain.CONTINUOUS] * p.n_variables,
        objective_sense=p.objective_sense,
        objective_offset=p.objective_offset,
        name=f'{name}-lp-relaxation',
        metadata={**p.metadata, 'm25_source_instance': name, 'm25_relaxation': True},
    )


def _run(p: LinearProblem, method: str):
    b = ScipyHighsLPBackend(method=method, time_limit_s=CUTOFF_S)
    t0 = perf_counter()
    r = execute(p, b)
    wall = perf_counter() - t0
    terminal = r.status in {PublicStatus.VALID_OPTIMAL, PublicStatus.INFEASIBLE, PublicStatus.UNBOUNDED}
    valid = (r.validation is not None and r.validation.valid) if r.x is not None else terminal
    return r, wall, terminal, bool(valid)


def _cost(sample: dict[str, object]) -> float:
    return min(float(sample['wall_s']), CUTOFF_S) if sample['terminal'] else CUTOFF_S


def _bootstrap_ratio(a: np.ndarray, b: np.ndarray, draws: int = BOOTSTRAP_DRAWS) -> dict[str, float | int]:
    rng = np.random.default_rng(250025)
    n = len(a)
    vals = np.empty(draws)
    for i in range(draws):
        idx = rng.integers(0, n, size=n)
        vals[i] = float(a[idx].mean() / b[idx].mean())
    q = np.quantile(vals, [0.025, 0.5, 0.975])
    return {
        'point': float(a.mean() / b.mean()),
        'low': float(q[0]),
        'median': float(q[1]),
        'high': float(q[2]),
        'draws': draws,
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument('--miplib-zip', type=Path, required=True)
    ap.add_argument('--manifest', type=Path, required=True)
    ap.add_argument('--output', type=Path, required=True)
    ap.add_argument('--m22-integrity', type=Path)
    ap.add_argument('--shard-index', type=int, default=0)
    ap.add_argument('--shard-count', type=int, default=1)
    args = ap.parse_args()

    names = [
        x.strip() for x in args.manifest.read_text(encoding='utf-8').splitlines()
        if x.strip() and not x.lstrip().startswith('#')
    ]
    zip_sha = _sha256(args.miplib_zip)
    integrity_verified = False
    integrity_source = None
    if args.m22_integrity is not None:
        integ = json.loads(args.m22_integrity.read_text(encoding='utf-8'))
        mip = (integ.get('archives') or {}).get('miplib') if isinstance(integ, dict) else None
        integrity_verified = bool(integ.get('all_integrity_pass')) and isinstance(mip, dict) and mip.get('sha256_match') is True and mip.get('sha256') == zip_sha
        integrity_source = str(args.m22_integrity.resolve())

    with zipfile.ZipFile(args.miplib_zip) as z:
        info = {Path(i.filename).name: i for i in z.infolist() if i.filename.endswith('.mps.gz')}
        pre = [n for n in names if n in info and info[n].compress_size <= MAX_COMPRESSED_BYTES]
        pre = sorted(pre, key=_filename_key)
        parse_exclusions: list[dict[str, object]] = []
        structural_exclusions: list[dict[str, object]] = []
        cohort: list[tuple[str, LinearProblem]] = []
        for name in pre:
            if len(cohort) >= TARGET_INSTANCES:
                break
            i = info[name]
            try:
                raw = z.read(i)
                text = gzip.decompress(raw).decode('utf-8', errors='strict')
                original = parse_mps(text)
                p = _relax(original, Path(name).name.removesuffix('.mps.gz'))
            except Exception as exc:
                parse_exclusions.append({'instance': name, 'error': f'{type(exc).__name__}: {exc}'})
                continue
            if p.n_variables > MAX_N or p.n_constraints > MAX_M or p.nnz > MAX_NNZ:
                structural_exclusions.append({'instance': name, 'n': p.n_variables, 'm': p.n_constraints, 'nnz': p.nnz})
                continue
            cohort.append((name, p))

    if args.shard_count < 1 or args.shard_index < 0 or args.shard_index >= args.shard_count:
        raise ValueError('invalid shard index/count')
    full_cohort_names = [name for name, _ in cohort]
    cohort_signature = hashlib.sha256('\n'.join(full_cohort_names).encode()).hexdigest()
    cohort = [(name,p) for idx,(name,p) in enumerate(cohort) if idx % args.shard_count == args.shard_index]

    # Warm both routes before measurements.
    warm = LinearProblem.from_data(
        A=[[1.0, 1.0]], c=[1.0, 2.0],
        variable_lower=[0.0, 0.0], variable_upper=[1.0, 1.0],
        constraint_lower=[1.0], constraint_upper=[np.inf],
    )
    for method in ('highs-ds', 'highs-ipm'):
        _run(warm, method)

    rows: list[dict[str, object]] = []
    started = perf_counter()
    for idx, (name, p) in enumerate(cohort):
        inspect_samples: list[float] = []
        samples = {'highs-ds': [], 'highs-ipm': []}
        for round_idx in range(ROUNDS):
            t0 = perf_counter()
            fp = inspect_problem(p)
            inspect_samples.append(perf_counter() - t0)
            order = ['highs-ds', 'highs-ipm']
            if (idx + round_idx) % 2:
                order.reverse()
            for method in order:
                r, wall, terminal, valid = _run(p, method)
                samples[method].append({
                    'wall_s': wall,
                    'status': r.status.value,
                    'objective': None if r.objective is None else float(r.objective),
                    'terminal': terminal,
                    'valid': valid,
                })

        ds = samples['highs-ds']
        ip = samples['highs-ipm']
        ds_cost = float(statistics.median(_cost(x) for x in ds))
        ip_cost = float(statistics.median(_cost(x) for x in ip))
        inspect_cost = float(statistics.median(inspect_samples))
        ds_opt = [x['objective'] for x in ds if x['status'] == 'valid_optimal' and x['objective'] is not None]
        ip_opt = [x['objective'] for x in ip if x['status'] == 'valid_optimal' and x['objective'] is not None]
        obj_diff = None
        if ds_opt and ip_opt:
            obj_diff = abs(float(statistics.median(ds_opt)) - float(statistics.median(ip_opt)))
        winner = 'tie'
        if ds_cost < ip_cost * 0.995:
            winner = 'highs-ds'
        elif ip_cost < ds_cost * 0.995:
            winner = 'highs-ipm'
        rows.append({
            'instance': name,
            'n': p.n_variables,
            'm': p.n_constraints,
            'nnz': p.nnz,
            'm_over_n': p.n_constraints / max(p.n_variables, 1),
            'density': p.nnz / max(p.n_variables * p.n_constraints, 1),
            'inspect_median_s': inspect_cost,
            'ds_median_cost_s': ds_cost,
            'ipm_median_cost_s': ip_cost,
            'vbs_cost_s': min(ds_cost, ip_cost),
            'oracle_with_inspection_cost_s': min(ds_cost, ip_cost) + inspect_cost,
            'winner': winner,
            'ipm_over_ds': ip_cost / max(ds_cost, 1e-15),
            'ds_terminal_rounds': sum(x['terminal'] for x in ds),
            'ipm_terminal_rounds': sum(x['terminal'] for x in ip),
            'ds_valid_rounds': sum(x['valid'] for x in ds),
            'ipm_valid_rounds': sum(x['valid'] for x in ip),
            'objective_abs_diff': obj_diff,
            'samples': samples,
        })

    if args.shard_count > 1:
        payload = {
            'schema': 'optimind.m25.public_ood_lp_opportunity_shard.v1',
            'source': 'MIPLIB 2017 benchmark-v2 derived continuous LP relaxations',
            'miplib_zip_sha256': zip_sha,
            'manifest_sha256': _sha256(args.manifest),
            'official_corpus_integrity_passed': integrity_verified,
            'integrity_source': integrity_source,
            'pre_registered_opportunity_gate': {
                'target_instances': TARGET_INSTANCES, 'rounds': ROUNDS, 'cutoff_s': CUTOFF_S,
                'min_relative_vbs_gain': MIN_RELATIVE_VBS_GAIN,
                'min_relative_oracle_feature_gain': MIN_RELATIVE_ORACLE_FEATURE_GAIN,
                'min_non_sbs_wins': MIN_NON_SBS_WINS,
                'min_decisive_non_sbs_wins': MIN_DECISIVE_NON_SBS_WINS,
                'decisive_ratio': DECISIVE_RATIO,
                'bootstrap_upper_must_be_below': 1.0, 'frozen_before_outcomes': True,
            },
            'selection_protocol': {
                'outcome_independent': True, 'compressed_size_cap_bytes': MAX_COMPRESSED_BYTES,
                'hash_order': 'sha256(filename)', 'target_instances': TARGET_INSTANCES,
                'max_n': MAX_N, 'max_m': MAX_M, 'max_nnz': MAX_NNZ,
            },
            'full_cohort_names': full_cohort_names,
            'cohort_signature': cohort_signature,
            'shard_index': args.shard_index, 'shard_count': args.shard_count,
            'rows': rows, 'parse_exclusions': parse_exclusions, 'structural_exclusions': structural_exclusions,
            'environment': {'python': os.sys.version, 'numpy': np.__version__, 'scipy': scipy.__version__, 'platform': platform.platform(), 'cpu_count': os.cpu_count()},
            'wall_s': perf_counter() - started,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
        print(json.dumps({'shard': args.shard_index, 'rows': len(rows), 'cohort_signature': cohort_signature, 'wall_s': payload['wall_s']}, indent=2))
        return

    ds = np.asarray([r['ds_median_cost_s'] for r in rows], dtype=float)
    ip = np.asarray([r['ipm_median_cost_s'] for r in rows], dtype=float)
    inspect = np.asarray([r['inspect_median_s'] for r in rows], dtype=float)
    means = {'scipy-highs-ds': float(ds.mean()), 'scipy-highs-ipm': float(ip.mean())}
    sbs_name = min(means, key=means.get)
    sbs = ds if sbs_name.endswith('ds') else ip
    vbs = np.minimum(ds, ip)
    oracle_feature = vbs + inspect
    vbs_gain = float((sbs.mean() - vbs.mean()) / sbs.mean()) if sbs.mean() else 0.0
    oracle_feature_gain = float((sbs.mean() - oracle_feature.mean()) / sbs.mean()) if sbs.mean() else 0.0
    boot = _bootstrap_ratio(oracle_feature, sbs)

    objective_mismatches = 0
    for r in rows:
        if r['objective_abs_diff'] is None:
            continue
        ref = next((x['objective'] for x in r['samples']['highs-ds'] if x['objective'] is not None), 0.0)
        tol = 1e-6 * max(1.0, abs(float(ref)))
        if float(r['objective_abs_diff']) > tol:
            objective_mismatches += 1

    total = len(rows) * ROUNDS
    ds_terminal = sum(int(r['ds_terminal_rounds']) for r in rows)
    ip_terminal = sum(int(r['ipm_terminal_rounds']) for r in rows)
    non_sbs_wins = sum(r['winner'] == 'highs-ipm' if sbs_name.endswith('ds') else r['winner'] == 'highs-ds' for r in rows)
    decisive_non_sbs_wins = 0
    if sbs_name.endswith('ds'):
        decisive_non_sbs_wins = sum(float(r['ipm_over_ds']) <= DECISIVE_RATIO for r in rows)
    else:
        decisive_non_sbs_wins = sum(float(r['ipm_over_ds']) >= 1.0 / DECISIVE_RATIO for r in rows)

    gates = {
        'official_corpus_integrity': integrity_verified,
        'minimum_instances': len(rows) >= TARGET_INSTANCES,
        'ds_terminal_rate_ge_0_90': ds_terminal / max(total, 1) >= 0.90,
        'ipm_terminal_rate_ge_0_90': ip_terminal / max(total, 1) >= 0.90,
        'objective_mismatches_zero': objective_mismatches == 0,
        'vbs_relative_gain_ge_0_03': vbs_gain >= MIN_RELATIVE_VBS_GAIN,
        'oracle_with_feature_relative_gain_ge_0_02': oracle_feature_gain >= MIN_RELATIVE_ORACLE_FEATURE_GAIN,
        'oracle_bootstrap_upper_below_1': float(boot['high']) < 1.0,
        'non_sbs_wins_ge_3': non_sbs_wins >= MIN_NON_SBS_WINS,
        'decisive_non_sbs_wins_ge_2': decisive_non_sbs_wins >= MIN_DECISIVE_NON_SBS_WINS,
    }
    opportunity_validated = bool(gates) and all(gates.values())

    payload = {
        'schema': 'optimind.m25.public_ood_lp_opportunity.v1',
        'source': 'MIPLIB 2017 benchmark-v2 derived continuous LP relaxations',
        'purpose': 'pre-selector opportunity audit; no selector is trained or promoted in M25',
        'miplib_zip_sha256': zip_sha,
        'manifest_sha256': _sha256(args.manifest),
        'official_corpus_integrity_passed': integrity_verified,
        'integrity_source': integrity_source,
        'pre_registered_opportunity_gate': {
            'target_instances': TARGET_INSTANCES,
            'rounds': ROUNDS,
            'cutoff_s': CUTOFF_S,
            'min_relative_vbs_gain': MIN_RELATIVE_VBS_GAIN,
            'min_relative_oracle_feature_gain': MIN_RELATIVE_ORACLE_FEATURE_GAIN,
            'min_non_sbs_wins': MIN_NON_SBS_WINS,
            'min_decisive_non_sbs_wins': MIN_DECISIVE_NON_SBS_WINS,
            'decisive_ratio': DECISIVE_RATIO,
            'bootstrap_upper_must_be_below': 1.0,
            'frozen_before_outcomes': True,
        },
        'selection_protocol': {
            'outcome_independent': True,
            'compressed_size_cap_bytes': MAX_COMPRESSED_BYTES,
            'hash_order': 'sha256(filename)',
            'target_instances': TARGET_INSTANCES,
            'max_n': MAX_N,
            'max_m': MAX_M,
            'max_nnz': MAX_NNZ,
        },
        'instances': len(rows),
        'parse_exclusions': parse_exclusions,
        'structural_exclusions': structural_exclusions,
        'rounds': ROUNDS,
        'cutoff_s': CUTOFF_S,
        'metrics': {
            'sbs_backend': sbs_name,
            'sbs_mean_s': float(sbs.mean()),
            'vbs_mean_s': float(vbs.mean()),
            'oracle_with_inspection_mean_s': float(oracle_feature.mean()),
            'vbs_relative_gain': vbs_gain,
            'oracle_with_inspection_relative_gain': oracle_feature_gain,
            'oracle_with_inspection_to_sbs_bootstrap': boot,
            'ds_wins': sum(r['winner'] == 'highs-ds' for r in rows),
            'ipm_wins': sum(r['winner'] == 'highs-ipm' for r in rows),
            'ties': sum(r['winner'] == 'tie' for r in rows),
            'non_sbs_wins': non_sbs_wins,
            'decisive_non_sbs_wins': decisive_non_sbs_wins,
            'objective_mismatches': objective_mismatches,
            'ds_terminal_rate': ds_terminal / max(total, 1),
            'ipm_terminal_rate': ip_terminal / max(total, 1),
            'median_inspection_s': float(statistics.median(inspect)) if len(inspect) else None,
        },
        'opportunity_gates': gates,
        'selection_opportunity_validated': opportunity_validated,
        'environment': {
            'python': os.sys.version,
            'numpy': np.__version__,
            'scipy': scipy.__version__,
            'platform': platform.platform(),
            'cpu_count': os.cpu_count(),
        },
        'wall_s': perf_counter() - started,
        'claims_boundary': [
            'M25 measures whether a nontrivial DS-vs-IPM selection opportunity exists after current inspection cost',
            'M25 does not train, tune, or authorize a production selector',
            'a failed opportunity gate means automatic LP performance routing remains disabled',
        ],
        'rows': rows,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding='utf-8')
    print(json.dumps({
        'instances': payload['instances'],
        'metrics': payload['metrics'],
        'opportunity_gates': gates,
        'selection_opportunity_validated': opportunity_validated,
        'wall_s': payload['wall_s'],
    }, indent=2, sort_keys=True))


if __name__ == '__main__':
    main()
