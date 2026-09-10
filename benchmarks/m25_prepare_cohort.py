from __future__ import annotations
import argparse,gzip,hashlib,json,pickle,zipfile
from pathlib import Path
from solverpilot import LinearProblem,VariableDomain,parse_mps

TARGET_INSTANCES=48; MAX_COMPRESSED_BYTES=512*1024; MAX_N=5000; MAX_M=5000; MAX_NNZ=100000

def key(name): return hashlib.sha256(name.encode()).hexdigest()
def sha(path):
 h=hashlib.sha256();
 with open(path,'rb') as f:
  for c in iter(lambda:f.read(1024*1024),b''): h.update(c)
 return h.hexdigest()
def relax(p,name):
 return LinearProblem.from_data(A=p.A,c=p.c,variable_lower=p.variable_lower,variable_upper=p.variable_upper,constraint_lower=p.constraint_lower,constraint_upper=p.constraint_upper,domains=[VariableDomain.CONTINUOUS]*p.n_variables,objective_sense=p.objective_sense,objective_offset=p.objective_offset,name=f'{name}-lp-relaxation',metadata={**p.metadata,'m25_source_instance':name,'m25_relaxation':True})

def main():
 ap=argparse.ArgumentParser(); ap.add_argument('--miplib-zip',type=Path,required=True); ap.add_argument('--manifest',type=Path,required=True); ap.add_argument('--cache-dir',type=Path,required=True); ap.add_argument('--output',type=Path,required=True); args=ap.parse_args()
 names=[x.strip() for x in args.manifest.read_text().splitlines() if x.strip() and not x.lstrip().startswith('#')]
 args.cache_dir.mkdir(parents=True,exist_ok=True); parse_ex=[]; struct_ex=[]; selected=[]
 with zipfile.ZipFile(args.miplib_zip) as z:
  info={Path(i.filename).name:i for i in z.infolist() if i.filename.endswith('.mps.gz')}
  pre=sorted([n for n in names if n in info and info[n].compress_size<=MAX_COMPRESSED_BYTES],key=key)
  for name in pre:
   if len(selected)>=TARGET_INSTANCES: break
   try:
    p=relax(parse_mps(gzip.decompress(z.read(info[name])).decode('utf-8','strict')),Path(name).name.removesuffix('.mps.gz'))
   except Exception as exc:
    parse_ex.append({'instance':name,'error':f'{type(exc).__name__}: {exc}'}); continue
   if p.n_variables>MAX_N or p.n_constraints>MAX_M or p.nnz>MAX_NNZ:
    struct_ex.append({'instance':name,'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz}); continue
   cache=args.cache_dir/(name+'.pkl'); cache.write_bytes(pickle.dumps(p,protocol=pickle.HIGHEST_PROTOCOL))
   selected.append({'instance':name,'cache':str(cache.resolve()),'n':p.n_variables,'m':p.n_constraints,'nnz':p.nnz})
 sig=hashlib.sha256('\n'.join(x['instance'] for x in selected).encode()).hexdigest()
 payload={'schema':'optimind.m25.prepared_cohort.v1','miplib_zip_sha256':sha(args.miplib_zip),'manifest_sha256':sha(args.manifest),'selection_protocol':{'outcome_independent':True,'compressed_size_cap_bytes':MAX_COMPRESSED_BYTES,'hash_order':'sha256(filename)','target_instances':TARGET_INSTANCES,'max_n':MAX_N,'max_m':MAX_M,'max_nnz':MAX_NNZ},'cohort_signature':sig,'instances':len(selected),'selected':selected,'parse_exclusions':parse_ex,'structural_exclusions':struct_ex}
 args.output.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps({'instances':len(selected),'signature':sig,'parse_exclusions':len(parse_ex),'structural_exclusions':len(struct_ex)},indent=2))
if __name__=='__main__': main()
