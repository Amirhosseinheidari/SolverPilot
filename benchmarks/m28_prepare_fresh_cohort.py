from __future__ import annotations

import argparse
import concurrent.futures
import hashlib
import json
import os
import subprocess
import sys
import zipfile
from pathlib import Path

TARGET_INSTANCES = 48
MAX_COMPRESSED_BYTES = 2 * 1024 * 1024
MAX_N = 12000
MAX_M = 12000
MAX_NNZ = 300000
PARSE_HARD_TIMEOUT_S = 8.0
PARSE_WORKERS = 8


def sha256_file(path: Path) -> str:
    h=hashlib.sha256()
    with path.open('rb') as f:
        for block in iter(lambda:f.read(1024*1024),b''): h.update(block)
    return h.hexdigest()

def order_key(name: str) -> str:
    return hashlib.sha256(name.encode('utf-8')).hexdigest()

def run_parse(worker: Path, root: Path, zip_path: Path, cache_dir: Path, name: str) -> dict:
    out=cache_dir/(name+'.pkl')
    if out.exists():
        try:
            import pickle
            p=pickle.loads(out.read_bytes())
            return {'instance':name,'ok':True,'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz,'cache':str(out.resolve()),'cache_sha256':sha256_file(out),'cache_reused':True}
        except Exception:
            out.unlink(missing_ok=True)
    env=dict(os.environ); roots=[str((root/'src').resolve())]; blocked=('/python-hooks','/pyvenv-overrides');
    for raw in sys.path:
        q=os.getcwd() if raw=='' else raw
        if not q or any(b in q for b in blocked): continue
        try: q=str(Path(q).resolve())
        except Exception: q=str(q)
        if Path(q).exists() and q not in roots: roots.append(q)
    env['PYTHONPATH']=os.pathsep.join(roots)
    try:
        proc=subprocess.run([sys.executable,'-S',str(worker),'--zip',str(zip_path),'--instance',name,'--output',str(out)],cwd=root,env=env,text=True,stdout=subprocess.PIPE,stderr=subprocess.PIPE,timeout=PARSE_HARD_TIMEOUT_S,check=False)
    except subprocess.TimeoutExpired:
        return {'instance':name,'ok':False,'reason':'parse_hard_timeout'}
    if proc.returncode!=0:
        return {'instance':name,'ok':False,'reason':'parse_error','stderr':proc.stderr[-1500:]}
    try:
        row=json.loads(proc.stdout.strip().splitlines()[-1]); row['ok']=True; row['cache']=str(out.resolve()); row['cache_sha256']=sha256_file(out); return row
    except Exception as exc:
        return {'instance':name,'ok':False,'reason':'worker_output_error','error':f'{type(exc).__name__}: {exc}'}

def main() -> None:
    ap=argparse.ArgumentParser(); ap.add_argument('--miplib-zip',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--prior-cohort',type=Path,required=True); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
    manifest_names=[x.strip() for x in args.manifest.read_text(encoding='utf-8').splitlines() if x.strip() and not x.lstrip().startswith('#')]
    prior=json.loads(args.prior_cohort.read_text(encoding='utf-8')); prior_names={x['instance'] for x in prior['selected']}
    if len(prior_names)!=48: raise SystemExit(f'expected prior M25 cohort of 48, got {len(prior_names)}')
    args.cache_dir.mkdir(parents=True,exist_ok=True)
    with zipfile.ZipFile(args.miplib_zip) as z:
        infos={Path(i.filename).name:i for i in z.infolist() if i.filename.endswith('.mps.gz')}
        candidates=sorted([n for n in manifest_names if n not in prior_names and n in infos and infos[n].compress_size<=MAX_COMPRESSED_BYTES],key=order_key)
    root=Path(__file__).resolve().parents[1]; worker=root/'benchmarks/m28_parse_worker.py'
    results={}
    with concurrent.futures.ThreadPoolExecutor(max_workers=PARSE_WORKERS) as ex:
        future_map={ex.submit(run_parse,worker,root,args.miplib_zip,args.cache_dir,n):n for n in candidates}
        for fut in concurrent.futures.as_completed(future_map):
            n=future_map[fut]
            try: results[n]=fut.result()
            except Exception as exc: results[n]={'instance':n,'ok':False,'reason':'controller_error','error':f'{type(exc).__name__}: {exc}'}
    selected=[]; exclusions=[]
    for name in candidates:
        row=results[name]
        if not row.get('ok'):
            exclusions.append(row); continue
        if row['n']>MAX_N or row['m']>MAX_M or row['nnz']>MAX_NNZ:
            exclusions.append({'instance':name,'ok':False,'reason':'structural_cap','n':row['n'],'m':row['m'],'nnz':row['nnz']}); continue
        if len(selected)<TARGET_INSTANCES:
            selected.append({'instance':name,'cache':row['cache'],'cache_sha256':row['cache_sha256'],'n':row['n'],'m':row['m'],'nnz':row['nnz']})
    if len(selected)!=TARGET_INSTANCES: raise SystemExit(f'expected {TARGET_INSTANCES} selected instances, got {len(selected)}')
    names=[x['instance'] for x in selected]; overlap=sorted(set(names)&prior_names)
    if overlap: raise SystemExit(f'fresh cohort overlaps M25: {overlap}')
    sig=hashlib.sha256('\n'.join(names).encode()).hexdigest()
    payload={
      'schema':'optimind.m28.fresh_public_ood_cohort.v2','source':'MIPLIB 2017 benchmark-v2 continuous LP relaxations',
      'miplib_zip_sha256':sha256_file(args.miplib_zip),'manifest_sha256':sha256_file(args.manifest),
      'prior_cohort_file_sha256':sha256_file(args.prior_cohort),'prior_cohort_signature':prior.get('cohort_signature'),'prior_cohort_instances':len(prior_names),'fresh_overlap_count':0,
      'selection_protocol':{'outcome_independent':True,'exclude_all_m25_instances':True,'compressed_size_cap_bytes':MAX_COMPRESSED_BYTES,'hash_order':'sha256(filename)','target_instances':TARGET_INSTANCES,'max_n':MAX_N,'max_m':MAX_M,'max_nnz':MAX_NNZ,'parse_process_isolated':True,'parse_hard_timeout_s':PARSE_HARD_TIMEOUT_S,'parse_workers':PARSE_WORKERS},
      'cohort_signature':sig,'instances':len(selected),'selected':selected,'candidate_count':len(candidates),'exclusions':exclusions,
    }
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(payload,indent=2,sort_keys=True),encoding='utf-8')
    print(json.dumps({'instances':len(selected),'cohort_signature':sig,'candidate_count':len(candidates),'exclusions':len(exclusions),'fresh_overlap_count':0},indent=2))
if __name__=='__main__': main()
