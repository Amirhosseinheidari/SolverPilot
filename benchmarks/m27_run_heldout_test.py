from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, pickle, statistics, subprocess, sys
from pathlib import Path
from time import perf_counter
import numpy as np
from solverpilot import inspect_problem
from solverpilot.experimental import decide_selective_lp_backend
from solverpilot.benchmark.environment import thread_environment
from solverpilot.benchmark.runner import _clean_pythonpath

ROUNDS=2; CUTOFF_S=1.0; HARD_TIMEOUT_S=3.0; BOOTSTRAP_DRAWS=20000; BOOTSTRAP_SEED=270027


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()
def sample(worker,cache,method,root,env):
    try:
        proc=subprocess.run([sys.executable,'-S',str(worker),'--problem',str(cache),'--method',method,'--cutoff-s',str(CUTOFF_S)],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=HARD_TIMEOUT_S,check=False)
    except subprocess.TimeoutExpired:
        return {'wall_s':HARD_TIMEOUT_S,'status':'hard_timeout','objective':None,'terminal':False,'valid':False,'controller_hard_timeout':True,'worker_error':False}
    if proc.returncode!=0:
        return {'wall_s':HARD_TIMEOUT_S,'status':'worker_error','objective':None,'terminal':False,'valid':False,'controller_hard_timeout':False,'worker_error':True,'stderr':proc.stderr[-1000:]}
    row=json.loads(proc.stdout.strip().splitlines()[-1]); row['wall_s']=float(row.pop('solve_s')); row['worker_error']=False; return row

def cost(x): return min(float(x['wall_s']),CUTOFF_S) if x['terminal'] else CUTOFF_S

def bootstrap(policy, sbs):
    rng=np.random.default_rng(BOOTSTRAP_SEED); n=len(policy); vals=np.empty(BOOTSTRAP_DRAWS)
    for i in range(BOOTSTRAP_DRAWS):
        idx=rng.integers(0,n,size=n); vals[i]=policy[idx].mean()/sbs[idx].mean()
    q=np.quantile(vals,[.025,.5,.975])
    return {'point':float(policy.mean()/sbs.mean()),'low':float(q[0]),'median':float(q[1]),'high':float(q[2]),'draws':BOOTSTRAP_DRAWS,'seed':BOOTSTRAP_SEED}

def main():
    ap=argparse.ArgumentParser(); ap.add_argument('--model',type=Path,required=True); ap.add_argument('--split',type=Path,required=True); ap.add_argument('--protocol',type=Path,required=True); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--workers',type=int,default=4); args=ap.parse_args()
    if args.output.exists(): raise SystemExit('held-out test result already exists; M27 test is one-time only')
    model=json.loads(args.model.read_text()); split=json.loads(args.split.read_text())
    if model.get('test_outcomes_seen_during_fit') is not False or not model.get('development_gate_passed'): raise SystemExit('model is not eligible for held-out test')
    if model.get('split_sha256')!=split.get('split_sha256'): raise SystemExit('split mismatch')
    if model.get('protocol_sha256')!=sha(args.protocol): raise SystemExit('protocol changed after model freeze')
    test=[r for r in split['records'] if r['split']=='test']
    if len(test)!=16 or len({r['group'] for r in test})>=17: raise SystemExit('invalid test split')
    root=Path(__file__).resolve().parents[1]; worker=root/'benchmarks/m25_solve_worker.py'; env=dict(os.environ); env.update(thread_environment(1)); env['PYTHONPATH']=_clean_pythonpath()

    prepared=[]
    for rec in test:
        cache=args.cache_dir/(rec['instance']+'.pkl')
        if not cache.exists(): raise SystemExit(f'missing prepared test cache: {cache}')
        p=pickle.loads(cache.read_bytes()); inspections=[]; fp=None
        for _ in range(ROUNDS):
            t0=perf_counter(); fp=inspect_problem(p); inspections.append(perf_counter()-t0)
        decision=decide_selective_lp_backend(fp,model)
        prepared.append({'instance':rec['instance'],'group':rec['group'],'cache':cache,'fingerprint':fp,'decision':decision,'inspect_median_s':float(statistics.median(inspections))})

    def run_item(item):
        samples={'highs-ds':[],'highs-ipm':[]}
        stable_idx=int(hashlib.sha256(item['instance'].encode()).hexdigest()[:8],16)
        for rd in range(ROUNDS):
            order=['highs-ds','highs-ipm']
            if (stable_idx+rd)%2: order.reverse()
            for method in order: samples[method].append(sample(worker,item['cache'],method,root,env))
        ds=float(statistics.median(cost(x) for x in samples['highs-ds'])); ip=float(statistics.median(cost(x) for x in samples['highs-ipm']))
        ds_opt=[x['objective'] for x in samples['highs-ds'] if x['status']=='valid_optimal' and x['objective'] is not None]
        ip_opt=[x['objective'] for x in samples['highs-ipm'] if x['status']=='valid_optimal' and x['objective'] is not None]
        diff=None
        if ds_opt and ip_opt: diff=abs(float(statistics.median(ds_opt))-float(statistics.median(ip_opt)))
        dec=item['decision']
        return {'instance':item['instance'],'group':item['group'],'n':item['fingerprint'].n_variables,'m':item['fingerprint'].n_constraints,'nnz':item['fingerprint'].nnz_a,'density':item['fingerprint'].density_a,'inspect_median_s':item['inspect_median_s'],'selected_backend':dec.backend,'switched':dec.switched,'selector_leaf':dec.leaf,'predicted_saving_s':dec.predicted_saving_s,'lower_saving_s':dec.lower_saving_s,'support_passed':dec.support_passed,'selector_reason':dec.reason,'ds_median_cost_s':ds,'ipm_median_cost_s':ip,'objective_abs_diff':diff,'samples':samples}

    started=perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.workers) as ex:
        rows=list(ex.map(run_item,prepared))
    rows=sorted(rows,key=lambda r:r['instance'])
    if len(rows)!=16 or len({r['instance'] for r in rows})!=16: raise SystemExit('incomplete test rows')
    outcomes=sum(len(r['samples'][m]) for r in rows for m in ('highs-ds','highs-ipm'))
    worker_errors=sum(bool(x.get('worker_error')) for r in rows for m in ('highs-ds','highs-ipm') for x in r['samples'][m])
    ds=np.asarray([r['ds_median_cost_s'] for r in rows],float); ip=np.asarray([r['ipm_median_cost_s'] for r in rows],float); inspect=np.asarray([r['inspect_median_s'] for r in rows],float)
    sbs_backend='scipy-highs-ds' if ds.mean()<=ip.mean() else 'scipy-highs-ipm'; sbs=ds if sbs_backend.endswith('ds') else ip
    default_backend=str(model['default_backend']); default=ds if default_backend.endswith('ds') else ip
    policy=np.asarray([(r['ds_median_cost_s'] if r['selected_backend'].endswith('ds') else r['ipm_median_cost_s'])+r['inspect_median_s'] for r in rows],float)
    boot=bootstrap(policy,sbs)
    mismatch=0
    for r in rows:
        if r['objective_abs_diff'] is None: continue
        ref=next((x['objective'] for x in r['samples']['highs-ds'] if x['objective'] is not None),0.0); tol=1e-6*max(1.0,abs(float(ref)))
        mismatch += float(r['objective_abs_diff'])>tol
    harmful=[]
    for r in rows:
        if not r['switched']: continue
        alt=r['ds_median_cost_s'] if model['alternate_backend'].endswith('ds') else r['ipm_median_cost_s']
        de=r['ds_median_cost_s'] if default_backend.endswith('ds') else r['ipm_median_cost_s']
        if alt>de: harmful.append({'instance':r['instance'],'solver_regret_s':float(alt-de)})
    metrics={'test_sbs_backend':sbs_backend,'test_sbs_mean_s':float(sbs.mean()),'model_default_backend':default_backend,'default_mean_s':float(default.mean()),'policy_mean_s':float(policy.mean()),'policy_to_sbs':float(policy.mean()/sbs.mean()),'bootstrap_policy_to_sbs':boot,'switches':sum(r['switched'] for r in rows),'switch_rate':float(np.mean([r['switched'] for r in rows])),'p90_policy_to_default_route':float(np.quantile(policy/np.maximum(default,1e-12),.90)),'max_harmful_switch_regret_s':float(max([x['solver_regret_s'] for x in harmful],default=0.0)),'harmful_switches':harmful,'objective_mismatches':int(mismatch),'worker_errors':int(worker_errors),'outcomes':int(outcomes),'expected_outcomes':16*2*ROUNDS,'hard_timeouts':sum(x['status']=='hard_timeout' for r in rows for m in ('highs-ds','highs-ipm') for x in r['samples'][m]),'median_inspection_s':float(statistics.median(inspect))}
    gates={'exact_16_test_instances':len(rows)==16,'complete_64_outcomes':outcomes==64,'worker_errors_zero':worker_errors==0,'objective_mismatches_zero':mismatch==0,'policy_to_sbs_le_0_97':metrics['policy_to_sbs']<=0.97,'bootstrap_upper_below_1':boot['high']<1.0,'p90_policy_to_default_le_1_25':metrics['p90_policy_to_default_route']<=1.25,'no_harmful_switch_over_0_10s':metrics['max_harmful_switch_regret_s']<=0.10,'at_least_2_switches':metrics['switches']>=2,'model_protocol_hashes_frozen':model['protocol_sha256']==sha(args.protocol)}
    passed=all(gates.values())
    payload={'schema':'optimind.m27.one_time_heldout_test.v1','source':'fresh process-isolated test over frozen 16 MIPLIB-derived LP relaxations','model_sha256':model['model_sha256'],'model_file_sha256':sha(args.model),'protocol_sha256':sha(args.protocol),'split_sha256':split['split_sha256'],'test_was_sealed_until_this_execution':True,'test_execution_is_one_time':True,'rounds':ROUNDS,'cutoff_s':CUTOFF_S,'hard_timeout_s':HARD_TIMEOUT_S,'controller_workers':args.workers,'solver_threads':1,'metrics':metrics,'promotion_gates':gates,'performance_ranking_validated':passed,'rows':rows,'wall_s':perf_counter()-started,'claims_boundary':['this is the sole M27 held-out promotion evaluation','no post-test model, threshold, feature, or split changes are permitted inside M27','passing authorizes only compatible continuous-LP DS-vs-IPM performance routing']}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps({'performance_ranking_validated':passed,'metrics':metrics,'gates':gates},indent=2))
    raise SystemExit(0 if passed else 3)
if __name__=='__main__': main()
