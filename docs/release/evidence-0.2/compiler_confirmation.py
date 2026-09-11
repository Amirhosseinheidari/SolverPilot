import os
for k in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS'):os.environ[k]='1'
import json,statistics,time
from pathlib import Path
import numpy as np
from solverpilot.model import Model

def make(kind,A,b,c):
    md=Model();x=md.variable(len(c),lower=0,upper=1)
    if kind=='objective':p=md.parameter(len(c),value=c);md.add(A@x>=b);md.minimize(p@x)
    elif kind=='localized_rhs':
        p=md.parameter(value=float(b[0]));md.add(A[0]@x>=p)
        for i in range(1,len(b)):md.add(A[i]@x>=float(b[i]))
        md.minimize(c@x)
    else:p=md.parameter(A.shape,value=A);md.add(p@x>=b);md.minimize(c@x)
    md.compile();return md,p

rows=[]
for kind in ('objective','localized_rhs','global_matrix'):
    for seed in (17,29,43):
        rng=np.random.default_rng(seed);A=rng.uniform(.1,1,(40,160));b=.25*A.sum(axis=1);c=rng.uniform(.1,2,160)
        cm,cp=make(kind,A,b,c);fm,fp=make(kind,A,b,c);ts={'cached':[],'full':[]}
        for i in range(11):
            value=c+rng.normal(0,.001,160) if kind=='objective' else float(b[0])+rng.normal(0,.01) if kind=='localized_rhs' else A+rng.normal(0,.0001,A.shape)
            cp.value=value;fp.value=value;r={}
            for k in (['cached','full'] if i%2 else ['full','cached']):
                t=time.perf_counter();r[k]=cm.compile() if k=='cached' else fm.compile(use_cache=False);dt=time.perf_counter()-t
                if i>=2:ts[k].append(dt*1000)
            assert r['cached'].execution_ir.data_hash==r['full'].execution_ir.data_hash
        med={k:statistics.median(v) for k,v in ts.items()}
        rows.append({'kind':kind,'seed':seed,'n':160,'m':40,'median_ms':med,'cached_over_full':med['cached']/med['full'],'samples_ms':ts,'identical_data_hash':True})
Path(__file__).with_name('compiler-confirmation.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
print([(r['kind'],r['seed'],round(r['cached_over_full'],3)) for r in rows])
