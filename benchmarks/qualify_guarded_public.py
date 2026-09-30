"""Frozen public calibration/evaluation of the total-cost guard; no test tuning."""
import argparse
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from time import perf_counter

import numpy as np
from solverpilot import read_mps
from solverpilot.benchmark.environment import capture_environment, thread_environment
from solverpilot.experimental.learned_lp import LPObservation, LPSelector, fit_lp_selector, lp_features
from solverpilot.experimental.lp_environment import bind_lp_environment, _fingerprint
from solverpilot.experimental.lp_gain import LPGainGuard, calibrate_gain_guard, implementation_id
from solverpilot.experimental.robust_lp import solve_robust_lp
from prepare_public_lp import relax
from public_lp_worker import CANDIDATES, PACKAGES, candidate, run_strategy
from qualify_public_lp import run_process, sha, write
from qualify_robust_lp import verify_split
from public_lp_families import audit_split

API_LIMIT=2.
PROCESS_LIMIT=12.
REPEATS=2
STRATEGIES=('guarded','default','production')


def worker(args):
    p=relax(read_mps(args.mps))
    if args.strategy!='guarded':
        result=run_strategy(p,args.strategy,args.model,API_LIMIT)
    else:
        setup=perf_counter()
        model=LPSelector.load(args.model)
        guard=LPGainGuard.load(args.model.with_name('guard.json'))
        binding=bind_lp_environment(model,args.model.with_name('protocol.json'),capture_environment(packages=PACKAGES))
        backends={name:candidate(name,API_LIMIT) for name in CANDIDATES}
        preparation=perf_counter()-setup
        # The API accepts preloaded model/guard/backend objects. Artifact loading
        # is outside its budget, but remains in full controller wall observations.
        start=perf_counter()
        try:
            r=solve_robust_lp(p,model,environment_id=binding['decision_environment_id'],
                             backends=backends,time_limit_s=API_LIMIT,gain_guard=guard)
            result={'status':r.status.value,'verified':r.optimality_evidence.independently_verified_optimal,
                    'primal_valid':bool(r.validation and r.validation.valid),'objective':r.objective,
                    'api_wall_s':perf_counter()-start,'route':dict(r.raw_statistics['experimental_lp_route'])}
        except TimeoutError:
            result={'status':'setup_timeout','verified':False,'api_wall_s':perf_counter()-start}
        result.update(artifact_preparation_s=preparation,environment_binding=binding)
    print(json.dumps({**result,'data_hash':p.data_hash,'features':lp_features(p)},allow_nan=False))


def success(row):
    return (row.get('verified') is True and row.get('api_wall_s',float('inf'))<=API_LIMIT
            and row['wall_s']<=PROCESS_LIMIT)


def summarize(rows,cases):
    expected={(c['file'],s,r) for c in cases for s in STRATEGIES for r in range(REPEATS)}
    actual=[(r['instance'],r['strategy'],r['repeat']) for r in rows]
    if len(actual)!=len(set(actual)) or set(actual)!=expected:raise ValueError('incomplete outcomes')
    costs={};counts={};mismatches=[]
    for c in cases:
        values=[r['objective'] for r in rows if r['instance']==c['file'] and r.get('verified')]
        reference=c.get('reference')
        if values and any(abs(v-values[0])>1e-6*max(1,abs(values[0])) for v in values):mismatches.append(c['file'])
        if reference is not None and any(abs(v-reference)>1e-6*max(1,abs(reference)) for v in values):mismatches.append(c['file']+':reference')
    for strategy in STRATEGIES:
        costs[strategy]=[];counts[strategy]=[]
        for c in cases:
            group=[r for r in rows if r['instance']==c['file'] and r['strategy']==strategy]
            costs[strategy].append(float(np.mean([r['api_wall_s'] if success(r) else 10*API_LIMIT for r in group])))
            counts[strategy].append(sum(success(r) for r in group))
    comparisons={}
    for strategy in STRATEGIES[1:]:
        policy=np.asarray(costs['guarded']);base=np.asarray(costs[strategy]);rng=np.random.default_rng(271828)
        # Cohort admission requires one distinct declared family per case.
        samples=rng.integers(0,len(cases),size=(2000,len(cases)))
        boot=policy[samples].sum(axis=1)/base[samples].sum(axis=1)
        comparisons[strategy]={'ratio':float(policy.sum()/base.sum()),
            'bootstrap_95pct':[float(np.quantile(boot,.025)),float(np.quantile(boot,.975))],
            'p90_ratio':float(np.quantile(policy/base,.9)),
            'no_lost_verified_solves':all(a>=b for a,b in zip(counts['guarded'],counts[strategy]))}
    switches=sum(r.get('route',{}).get('candidate','production')!='production' for r in rows if r['strategy']=='guarded')
    gates={'complete_outcomes':True,'no_objective_mismatch':not mismatches,
           'at_least_20_groups':len({c['group'] for c in cases})>=20,'actual_guarded_switches':switches>=2}
    for name,value in comparisons.items():
        gates.update({name+'_gain':value['ratio']<=.97,name+'_bootstrap':value['bootstrap_95pct'][1]<1,
                      name+'_tail':value['p90_ratio']<=1.25,name+'_successes':value['no_lost_verified_solves']})
    return {'instances':len(cases),'outcomes':len(rows),'verified_within_budget':{k:sum(v) for k,v in counts.items()},
            'mean_api_par10_s':{k:float(np.mean(v)) for k,v in costs.items()},'comparisons':comparisons,
            'switches':switches,'objective_mismatches':mismatches,'gates':gates,
            'public_gate_passed':all(gates.values()),'automatic_production_routing_enabled':False}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--training',type=Path);parser.add_argument('--heldout',type=Path)
    parser.add_argument('--output',type=Path);parser.add_argument('--mps',type=Path)
    parser.add_argument('--model',type=Path);parser.add_argument('--strategy',choices=(*STRATEGIES,*CANDIDATES))
    args=parser.parse_args()
    if args.mps is not None:return worker(args)
    if os.name!='posix' or not hasattr(os,'sched_setaffinity'):raise SystemExit('Linux/WSL required')
    train=json.loads((args.training/'cohort.json').read_text());test=json.loads((args.heldout/'cohort.json').read_text())
    verify_split(train['selected'],test['selected'])
    family_audit=audit_split(train['selected'],test['selected'],
                            prior_names=test.get('prior_audit',{}).get('names',()))
    if not family_audit['passed']:
        raise ValueError('public family audit rejected split: '+json.dumps(family_audit))
    for folder,cohort in ((args.training,train),(args.heldout,test)):
        for c in cohort['selected']:
            if sha(folder/c['file'])!=c['mps_sha256']:raise ValueError('corpus changed')
    args.output.mkdir(parents=True,exist_ok=False)
    cpu=min(os.sched_getaffinity(0));os.sched_setaffinity(0,{cpu})
    env=dict(os.environ);env.update(thread_environment(1))
    if any(os.environ.get(k)!=v for k,v in thread_environment(1).items()):raise ValueError('parent thread limits must be one')
    environment=capture_environment(packages=PACKAGES);env_id=_fingerprint(environment)
    protocol={'schema':'solverpilot.robust-public-lp.v1','experiment':'total-cost gain guard',
        'cpu_only':True,'environment':environment,'captured_at':datetime.now(timezone.utc).isoformat(),
        'affinity_cpu':cpu,'implementation_sha256':implementation_id(),'runner_sha256':sha(__file__),
        'training_cohort_sha256':sha(args.training/'cohort.json'),'heldout_cohort_sha256':sha(args.heldout/'cohort.json'),
        'api_budget_s':API_LIMIT,'controller_budget_s':PROCESS_LIMIT,'repeats':REPEATS,
        'guard_overhead_s':.005,'cost':'full API time including routing/solver/verification; failed or late = 20 seconds',
        'artifact_preparation':'outside API; retained inside full controller wall measurement',
        'gate':'24 unseen cases/families; both baselines: >=3% gain, bootstrap upper<1, p90<=1.25, no lost successes; actual switches, zero objective mismatches',
        'retuning_allowed':False,'automatic_production_routing_enabled':False}
    write(args.output/'protocol.json',protocol);write(args.output/'training-cohort.json',train);write(args.output/'heldout-cohort.json',test)
    write(args.output/'family-audit.json',family_audit)
    model_path=args.output/'model.json'
    def collect(folder,cases,strategies,name):
        rows=[]
        for i,c in enumerate(cases):
            for repeat in range(REPEATS):
                shift=i%len(strategies);order=list(strategies[shift:]+strategies[:shift])
                if repeat:order.reverse()
                for strategy in order:
                    command=[sys.executable,str(Path(__file__).resolve()),'--mps',str(folder/c['file']),
                             '--strategy',strategy,'--model',str(model_path)]
                    result=run_process(command,env,PROCESS_LIMIT)
                    if result.get('data_hash',c['data_hash'])!=c['data_hash']:raise ValueError('model identity changed')
                    rows.append({'instance':c['file'],'strategy':strategy,'repeat':repeat,**result})
                    write(args.output/name,rows)
            print(json.dumps({'phase':name,'completed':c['file'],'outcomes':len(rows)}),flush=True)
        return rows
    training=collect(args.training,train['selected'],(*CANDIDATES,'production'),'training-observations.json')
    rows=[]
    for c in train['selected']:
        features=lp_features(relax(read_mps(args.training/c['file'])))
        outcomes=[r for r in training if r['instance']==c['file']]
        verified_values=[r['objective'] for r in outcomes if r.get('verified')]
        reference=c.get('reference')
        if verified_values and any(abs(v-verified_values[0])>1e-6*max(1,abs(verified_values[0])) for v in verified_values):
            raise ValueError('training verified objective mismatch')
        if reference is not None and any(abs(v-reference)>1e-6*max(1,abs(reference)) for v in verified_values):
            raise ValueError('training published-reference mismatch')
        samples={s:tuple((r.get('api_wall_s',10*API_LIMIT),success(r)) for r in outcomes if r['strategy']==s) for s in (*CANDIDATES,'production')}
        rows.append(LPObservation(c['file'],c['group'],c['data_hash'],'train',env_id,features,samples))
    fit_rows=[LPObservation(r.instance,r.group,r.data_hash,r.split,r.environment_id,r.features,{s:r.samples[s] for s in CANDIDATES}) for r in rows]
    model=fit_lp_selector(fit_rows,candidates=CANDIDATES,cutoff_s=API_LIMIT,protocol_sha256=sha(args.output/'protocol.json'))
    guard=calibrate_gain_guard(model,rows,overhead_limit_s=.005)
    model.save(model_path);write(args.output/'guard.json',guard.payload());write(args.output/'training-rows.json',[asdict(r) for r in rows])
    write(args.output/'freeze.json',{'model_sha256':sha(model_path),'guard_sha256':sha(args.output/'guard.json'),
                                   'before_any_heldout_solve':True,'captured_at':datetime.now(timezone.utc).isoformat()})
    heldout=collect(args.heldout,test['selected'],STRATEGIES,'heldout-observations.json')
    report=summarize(heldout,test['selected']);write(args.output/'summary.json',report)
    print(json.dumps(report),flush=True)


if __name__=='__main__':main()
