from __future__ import annotations
import argparse, concurrent.futures, hashlib, json, os, subprocess, sys
from pathlib import Path

HARD_TIMEOUT_S=6.0; WORKERS=8

def sha256_file(p:Path)->str:
 h=hashlib.sha256();
 with p.open('rb') as f:
  for b in iter(lambda:f.read(1<<20),b''):h.update(b)
 return h.hexdigest()

def run(worker:Path,root:Path,row:dict)->dict:
 env=dict(os.environ); env['OMP_NUM_THREADS']='1'; env['OPENBLAS_NUM_THREADS']='1'; env['MKL_NUM_THREADS']='1'; env['NUMEXPR_NUM_THREADS']='1'
 roots=[str((root/'src').resolve())]
 for raw in sys.path:
  q=os.getcwd() if raw=='' else raw
  if not q or '/python-hooks' in q or '/pyvenv-overrides' in q: continue
  try:q=str(Path(q).resolve())
  except Exception:q=str(q)
  if Path(q).exists() and q not in roots: roots.append(q)
 env['PYTHONPATH']=os.pathsep.join(roots)
 try:
  p=subprocess.run([sys.executable,'-S',str(worker),'--cache',row['cache'],'--instance',row['instance']],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=HARD_TIMEOUT_S,check=False)
 except subprocess.TimeoutExpired:
  return {'instance':row['instance'],'ok':False,'reason':'feature_hard_timeout'}
 if p.returncode!=0:
  return {'instance':row['instance'],'ok':False,'reason':'feature_worker_error','stderr':p.stderr[-2000:]}
 try:
  d=json.loads(p.stdout.strip().splitlines()[-1]); d['ok']=True; return d
 except Exception as exc:return {'instance':row['instance'],'ok':False,'reason':'feature_output_error','error':f'{type(exc).__name__}: {exc}'}

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--m25-cohort',type=Path,required=True); ap.add_argument('--m28-cohort',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 root=Path(__file__).resolve().parents[1]; worker=root/'benchmarks/m29_feature_worker.py'
 cohorts={}
 for name,p in [('m25',args.m25_cohort),('m28',args.m28_cohort)]:
  d=json.loads(p.read_text()); cohorts[name]=d['selected']
 allrows=[dict(x,cohort=c) for c,rows in cohorts.items() for x in rows]
 results=[]
 with concurrent.futures.ThreadPoolExecutor(max_workers=WORKERS) as ex:
  futs={ex.submit(run,worker,root,r):(r['cohort'],r['instance']) for r in allrows}
  for fut in concurrent.futures.as_completed(futs):
   c,i=futs[fut]
   try:d=fut.result()
   except Exception as exc:d={'instance':i,'ok':False,'reason':'controller_error','error':f'{type(exc).__name__}: {exc}'}
   d['cohort']=c; results.append(d)
 results=sorted(results,key=lambda x:(x['cohort'],x['instance']))
 payload={'schema':'optimind.m29.feature_extraction.v1','claim_boundary':'post-outcome representation audit; never promotion evidence','worker_sha256':sha256_file(worker),'protocol_sha256':sha256_file(root/'docs/research/M29-VALUE-OF-INFORMATION-PROTOCOL.md'),'hard_timeout_s':HARD_TIMEOUT_S,'workers':WORKERS,'rows':results,'counts':{c:{'total':sum(r['cohort']==c for r in results),'ok':sum(r['cohort']==c and r.get('ok') for r in results),'probe_ok':sum(r['cohort']==c and r.get('ok') and not r.get('probe_error') for r in results)} for c in cohorts}}
 args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8'); print(json.dumps(payload['counts'],indent=2))
if __name__=='__main__':main()
