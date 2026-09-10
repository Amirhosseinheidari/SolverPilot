from __future__ import annotations
import argparse,json,os,pickle,statistics,subprocess,sys
from pathlib import Path
from time import perf_counter
from solverpilot import inspect_problem
from solverpilot.benchmark.runner import _clean_pythonpath
from solverpilot.benchmark.environment import thread_environment

ROUNDS=2; CUTOFF_S=1.0; HARD_TIMEOUT_S=3.0

def sample(worker,cache,method,root,env):
 try:
  proc=subprocess.run([sys.executable,'-S',str(worker),'--problem',str(cache),'--method',method,'--cutoff-s',str(CUTOFF_S)],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=HARD_TIMEOUT_S,check=False)
 except subprocess.TimeoutExpired:
  return {'wall_s':HARD_TIMEOUT_S,'status':'hard_timeout','objective':None,'terminal':False,'valid':False,'controller_hard_timeout':True}
 if proc.returncode!=0:
  return {'wall_s':HARD_TIMEOUT_S,'status':'worker_error','objective':None,'terminal':False,'valid':False,'stderr':proc.stderr[-1000:]}
 row=json.loads(proc.stdout.strip().splitlines()[-1]); row['wall_s']=float(row.pop('solve_s')); return row

def cost(x): return min(float(x['wall_s']),CUTOFF_S) if x['terminal'] else CUTOFF_S

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--cohort',type=Path,required=True); ap.add_argument('--shard-index',type=int,required=True); ap.add_argument('--shard-count',type=int,required=True); ap.add_argument('--output',type=Path,required=True); ap.add_argument('--m22-integrity',type=Path); args=ap.parse_args()
 c=json.loads(args.cohort.read_text()); full=c['selected']; chosen=[r for i,r in enumerate(full) if i%args.shard_count==args.shard_index]
 root=Path(__file__).resolve().parents[1]; worker=root/'benchmarks/m25_solve_worker.py'; env=dict(os.environ); env.update(thread_environment(1)); env['PYTHONPATH']=_clean_pythonpath()
 integ_ok=False
 if args.m22_integrity:
  integ=json.loads(args.m22_integrity.read_text()); mip=(integ.get('archives') or {}).get('miplib'); integ_ok=bool(integ.get('all_integrity_pass')) and isinstance(mip,dict) and mip.get('sha256_match') is True and mip.get('sha256')==c['miplib_zip_sha256']
 rows=[]; started=perf_counter()
 for local_idx,item in enumerate(chosen):
  p=pickle.loads(Path(item['cache']).read_bytes()); inspections=[]; samples={'highs-ds':[],'highs-ipm':[]}
  global_idx=next(i for i,x in enumerate(full) if x['instance']==item['instance'])
  for rd in range(ROUNDS):
   t0=perf_counter(); inspect_problem(p); inspections.append(perf_counter()-t0)
   order=['highs-ds','highs-ipm'];
   if (global_idx+rd)%2: order.reverse()
   for method in order: samples[method].append(sample(worker,Path(item['cache']),method,root,env))
  ds=samples['highs-ds']; ip=samples['highs-ipm']; ds_cost=float(statistics.median(cost(x) for x in ds)); ip_cost=float(statistics.median(cost(x) for x in ip)); inspect_cost=float(statistics.median(inspections));
  ds_opt=[x['objective'] for x in ds if x['status']=='valid_optimal' and x['objective'] is not None]; ip_opt=[x['objective'] for x in ip if x['status']=='valid_optimal' and x['objective'] is not None]; diff=None
  if ds_opt and ip_opt: diff=abs(float(statistics.median(ds_opt))-float(statistics.median(ip_opt)))
  winner='tie'
  if ds_cost<ip_cost*.995: winner='highs-ds'
  elif ip_cost<ds_cost*.995: winner='highs-ipm'
  rows.append({'instance':item['instance'],'n':item['n'],'m':item['m'],'nnz':item['nnz'],'m_over_n':item['m']/max(item['n'],1),'density':item['nnz']/max(item['n']*item['m'],1),'inspect_median_s':inspect_cost,'ds_median_cost_s':ds_cost,'ipm_median_cost_s':ip_cost,'vbs_cost_s':min(ds_cost,ip_cost),'oracle_with_inspection_cost_s':min(ds_cost,ip_cost)+inspect_cost,'winner':winner,'ipm_over_ds':ip_cost/max(ds_cost,1e-15),'ds_terminal_rounds':sum(x['terminal'] for x in ds),'ipm_terminal_rounds':sum(x['terminal'] for x in ip),'ds_valid_rounds':sum(x['valid'] for x in ds),'ipm_valid_rounds':sum(x['valid'] for x in ip),'objective_abs_diff':diff,'samples':samples})
 payload={'schema':'optimind.m25.public_ood_lp_opportunity_shard.v2','source':'MIPLIB 2017 benchmark-v2 derived continuous LP relaxations','miplib_zip_sha256':c['miplib_zip_sha256'],'manifest_sha256':c['manifest_sha256'],'official_corpus_integrity_passed':integ_ok,'pre_registered_opportunity_gate':{'target_instances':48,'rounds':2,'cutoff_s':1.0,'min_relative_vbs_gain':0.03,'min_relative_oracle_feature_gain':0.02,'min_non_sbs_wins':3,'min_decisive_non_sbs_wins':2,'decisive_ratio':0.90,'bootstrap_upper_must_be_below':1.0,'frozen_before_outcomes':True},'selection_protocol':c['selection_protocol'],'full_cohort_names':[x['instance'] for x in full],'cohort_signature':c['cohort_signature'],'shard_index':args.shard_index,'shard_count':args.shard_count,'rows':rows,'environment':{'python':sys.version},'wall_s':perf_counter()-started}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps({'shard':args.shard_index,'rows':len(rows),'wall_s':payload['wall_s'],'timeouts':sum(x['status']=='hard_timeout' for r in rows for m in r['samples'].values() for x in m)},indent=2))
if __name__=='__main__': main()
