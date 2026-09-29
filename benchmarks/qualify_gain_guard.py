"""Fresh synthetic development qualification; never enables production routing."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter

import numpy as np
from scipy import sparse
from solverpilot import LinearProblem, solve, solve_production, SolveBudget
from solverpilot.backends import ScipyHighsLPBackend
from solverpilot.benchmark.environment import capture_environment
from solverpilot.experimental.learned_lp import LPObservation, fit_lp_selector, lp_features, _digest
from solverpilot.experimental.lp_gain import calibrate_gain_guard
from solverpilot.experimental.robust_lp import solve_robust_lp

CANDIDATES = ('simplex', 'ipm')


def problem(seed, index):
    rng = np.random.default_rng(seed)
    n = (80, 240, 720)[index % 3]
    a = sparse.random(n//2, n, density=6/n, random_state=rng, format='csr')
    return LinearProblem.from_data(A=a, c=-rng.uniform(.1, 2, n),
        variable_lower=np.zeros(n), variable_upper=np.ones(n),
        constraint_lower=np.full(n//2, -np.inf), constraint_upper=a@rng.uniform(.2, .7, n))


def run(p, strategy, cutoff, model=None, guard=None, environment=None):
    start = perf_counter()
    if strategy == 'production': result, _ = solve_production(p, budget=SolveBudget(wall_time_s=cutoff))
    elif strategy == 'guarded':
        result = solve_robust_lp(p, model, environment_id=environment, time_limit_s=cutoff,
            gain_guard=guard, backends={c: ScipyHighsLPBackend(method='highs-ds' if c=='simplex' else 'highs-ipm') for c in CANDIDATES})
    else:
        result = solve(p, backend=ScipyHighsLPBackend(method='highs-ds' if strategy=='simplex' else 'highs-ipm'),
                       budget=SolveBudget(wall_time_s=cutoff))
    elapsed = perf_counter()-start
    return {'wall_s': elapsed, 'verified': result.optimality_evidence.independently_verified_optimal,
            'route': dict(result.raw_statistics.get('experimental_lp_route', {}))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args=parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    env=capture_environment(packages=('numpy','scipy','highspy'))
    environment=_digest(env)
    protocol={'scope': __doc__, 'train_seeds': list(range(731100,731124)),
              'test_seeds':list(range(942200,942212)), 'cutoff_s':2., 'repeats':2,
              'overhead_limit_s':.005, 'min_groups':4,
              'family':'bounded random sparse packing LP; groups are seeds, not independent application families'}
    def save(name, data):
        (args.output/name).write_bytes(json.dumps(data,indent=2,allow_nan=False).encode())
    save('protocol.json', protocol)
    records=[]; rows=[]
    for i, seed in enumerate(protocol['train_seeds']):
        p=problem(seed,i); samples={c:[] for c in (*CANDIDATES,'production')}
        for repeat in range(2):
            order=tuple(samples) if repeat==0 else tuple(reversed(samples))
            for strategy in order:
                result=run(p,strategy,2.)
                samples[strategy].append((result['wall_s'],result['verified']))
                records.append(dict(split='train',seed=seed,repeat=repeat,strategy=strategy,**result))
        rows.append(LPObservation(str(seed),str(seed),p.data_hash,'train',environment,lp_features(p),samples))
    fit_rows=[LPObservation(r.instance,r.group,r.data_hash,r.split,r.environment_id,r.features,
                           {c:r.samples[c] for c in CANDIDATES}) for r in rows]
    model=fit_lp_selector(fit_rows,candidates=CANDIDATES,cutoff_s=2.,protocol_sha256=_digest(protocol))
    guard=calibrate_gain_guard(model,rows,overhead_limit_s=.005)
    # Freeze all training and calibration artifacts before any held-out solve.
    save('model.json',model.payload());save('guard.json',guard.payload())
    save('training.json',[asdict(r) for r in rows])
    heldout_hashes=[]
    for i,seed in enumerate(protocol['test_seeds']):
        p=problem(seed,i);heldout_hashes.append(p.data_hash)
        assert p.data_hash not in model.training_hashes
        for repeat in range(2):
            for strategy in (('guarded','production') if repeat==0 else ('production','guarded')):
                result=run(p,strategy,2.,model,guard,environment)
                records.append(dict(split='test',seed=seed,repeat=repeat,strategy=strategy,**result))
    totals={}
    for strategy in ('guarded','production'):
        outcomes=[r for r in records if r['split']=='test' and r['strategy']==strategy]
        totals[strategy]={'calls':len(outcomes),'verified_within_budget':sum(r['verified'] and r['wall_s']<=2 for r in outcomes),
            'par10_mean_s':float(np.mean([r['wall_s'] if r['verified'] and r['wall_s']<=2 else 20 for r in outcomes]))}
    summary={'scope': __doc__, 'environment':env,'totals':totals,'allowed_candidates':guard.candidate_gains,
             'heldout_hashes':heldout_hashes,'production_authorized':False,
             'public_heldout_qualification':'not performed for this guard'}
    save('observations.json',records);save('summary.json',summary)
    print(json.dumps({'totals':totals,'allowed_candidates':guard.candidate_gains}))


if __name__=='__main__':main()
