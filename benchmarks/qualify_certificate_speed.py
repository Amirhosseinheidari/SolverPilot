"""Paired verifier microbenchmark; no claim about full-solve or industrial speed."""
import argparse
from dataclasses import asdict
from fractions import Fraction
import hashlib
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from solverpilot import LinearProblem
import solverpilot.validate.optimality as verifier
from solverpilot.benchmark.environment import capture_environment


def reference_dot(a, b):
    return sum((Fraction(float(x))*Fraction(float(y)) for x,y in zip(a,b)), Fraction())


def reference_matvec(a, b):
    a=a.tocsr(); values=[Fraction(float(v)) for v in b]
    return [sum((Fraction(float(a.data[k]))*values[a.indices[k]]
            for k in range(a.indptr[i],a.indptr[i+1])), Fraction()) for i in range(a.shape[0])]


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--output',type=Path,required=True);args=ap.parse_args()
    if args.output.exists():raise SystemExit('refusing to overwrite evidence')
    old_dot,old_matvec=verifier.dot,verifier.matvec
    observations=[]
    for n in (500,5000,20000):
        rng=np.random.default_rng(20261000+n)
        a=sparse.random(n//2,n,density=4/n,random_state=rng,format='csr',data_rvs=lambda size:rng.uniform(-1,1,size))
        x=rng.uniform(.1,.9,n);y=rng.uniform(-1,1,n//2);activity=a@x
        p=LinearProblem.from_data(A=a,c=-a.T@y,variable_lower=np.zeros(n),variable_upper=np.ones(n),
            constraint_lower=np.where(y<0,activity,activity-1),constraint_upper=np.where(y>0,activity,activity+1))
        dual=np.r_[y,np.zeros(n)]
        checks={}
        for repeat in range(3):
            for method in (('reference','dyadic') if repeat%2==0 else ('dyadic','reference')):
                verifier.dot,verifier.matvec=(reference_dot,reference_matvec) if method=='reference' else (old_dot,old_matvec)
                try:
                    start=perf_counter();check=verifier.verify_optimality(p,x,dual);wall=perf_counter()-start
                finally:verifier.dot,verifier.matvec=old_dot,old_matvec
                checks[method]=asdict(check)
                observations.append({'n':n,'nnz':p.A.nnz,'repeat':repeat,'method':method,'wall_s':wall,'check':asdict(check)})
            assert checks['reference']==checks['dyadic'] and checks['dyadic']['verified']
    result={'scope':'verifier-only, planted bounded LP; exact same candidate, alternated order; three repeats; not full solve speed',
        'environment':capture_environment(packages=('numpy','scipy')),
        'source_sha256':hashlib.sha256(Path(verifier.__file__).read_bytes()).hexdigest(),
        'observations':observations,
        'median_ratio_by_n':{str(n):float(np.median([r['wall_s'] for r in observations if r['n']==n and r['method']=='dyadic'])/
            np.median([r['wall_s'] for r in observations if r['n']==n and r['method']=='reference'])) for n in (500,5000,20000)}}
    args.output.write_text(json.dumps(result,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(result['median_ratio_by_n']))


if __name__=='__main__':main()
