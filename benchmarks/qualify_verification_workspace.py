"""Paired complete numerical verifier timing; no production routing authority."""
import argparse
from dataclasses import asdict
import json
from pathlib import Path
from time import perf_counter
import numpy as np
from scipy import sparse
from solverpilot import LinearProblem
from solverpilot._identity import process_source_sha256
from solverpilot.benchmark.environment import capture_environment
from solverpilot.validate.optimality import verify_optimality
from solverpilot.validate.workspace import VerificationWorkspace


def main():
    parser=argparse.ArgumentParser(); parser.add_argument('--output',type=Path,required=True); args=parser.parse_args()
    if args.output.exists(): raise SystemExit('refusing to overwrite prior evidence')
    observations=[]
    for n in (500,5000,20000):
        rng=np.random.default_rng(20261014+n)
        A=sparse.random(n//2,n,density=4/n,random_state=rng,format='csr',data_rvs=lambda count:rng.uniform(-1,1,count))
        x=rng.uniform(.1,.9,n); y=rng.uniform(-1,1,n//2); activity=A@x
        problem=LinearProblem.from_data(A=A,c=-A.T@y,variable_lower=np.zeros(n),variable_upper=np.ones(n),
            constraint_lower=np.where(y<0,activity,activity-1),constraint_upper=np.where(y>0,activity,activity+1))
        dual=np.r_[y,np.zeros(n)]; workspace=VerificationWorkspace()
        for repeat in range(4):
            outcomes={}
            for method in (('uncached','workspace') if repeat%2==0 else ('workspace','uncached')):
                start=perf_counter(); check=verify_optimality(problem,x,dual,workspace=workspace if method=='workspace' else None)
                elapsed=perf_counter()-start; outcomes[method]=asdict(check)
                observations.append({'n':n,'nnz':problem.A.nnz,'data_hash':problem.data_hash,'repeat':repeat,'method':method,
                                     'elapsed_s':elapsed,'verified':check.verified,'dual_bound':check.dual_bound,
                                     'workspace':workspace.info() if method=='workspace' else None})
            assert outcomes['uncached']==outcomes['workspace'] and outcomes['workspace']['verified']
        assert not verify_optimality(problem,np.zeros(n),dual,workspace=workspace).verified
    ratios={str(n):float(np.median([r['elapsed_s'] for r in observations if r['n']==n and r['method']=='workspace'])/
                         np.median([r['elapsed_s'] for r in observations if r['n']==n and r['method']=='uncached'])) for n in (500,5000,20000)}
    report={'schema':'solverpilot.verification-workspace-experiment.v1','scope':'full numerical verifier including fresh primal validation; seeded bounded planted LPs; four alternating repeats; first cache preparation included; not native solver, public dataset or industrial performance qualification',
            'source_sha256':process_source_sha256(),'environment':capture_environment(packages=('numpy','scipy')),
            'observations':observations,'median_workspace_over_uncached':ratios,'production_authorized':False}
    args.output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8')
    print(json.dumps(ratios))


if __name__=='__main__': main()
