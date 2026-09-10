from __future__ import annotations
import json, os, platform, sys
from pathlib import Path
from statistics import median
from time import perf_counter
import numpy as np, scipy, casadi
from solverpilot import QuadraticProblem, execute
from solverpilot.backends import CasadiOSQPBridgeBackend
OUT=Path(__file__).parent/'results'/'m10-osqp-stateful-qp.json'
SIZES=(20,60,120); STEPS=15; REPEATS=5

def make_base(n, seed):
    rng=np.random.default_rng(seed)
    M=rng.normal(size=(n,n))/np.sqrt(n)
    P=M.T@M+np.eye(n)*0.2
    m=max(1,n//3); A=rng.normal(size=(m,n)); A/=np.maximum(np.linalg.norm(A,axis=1,keepdims=True),1e-12)
    x0=rng.uniform(-.3,.3,size=n); b=A@x0
    lo=b-1.0; hi=b+1.0
    return P,A,lo,hi,x0,rng

def problem(P,A,lo,hi,target):
    q=-(P@target)
    n=len(q)
    return QuadraticProblem.from_data(P=P,A=A,q=q,variable_lower=np.full(n,-2.),variable_upper=np.full(n,2.),constraint_lower=lo,constraint_upper=hi,verify_convexity=False)

def main():
    rows=[]; start=perf_counter()
    for n in SIZES:
      for rep in range(REPEATS):
        P,A,lo,hi,x0,rng=make_base(n,130000+n*10+rep)
        targets=[x0+0.03*rng.normal(size=n) for _ in range(STEPS)]
        # warm library init outside measured sequence
        execute(problem(P,A,lo,hi,targets[0]),CasadiOSQPBridgeBackend())
        state=CasadiOSQPBridgeBackend(); t0=perf_counter(); sr=[]; reuse=[]
        for t in targets:
          r=execute(problem(P,A,lo,hi,t),state); sr.append(r); reuse.append(r.trace.reuse_applied)
        state_s=perf_counter()-t0
        t0=perf_counter(); cr=[]
        for t in targets: cr.append(execute(problem(P,A,lo,hi,t),CasadiOSQPBridgeBackend()))
        cold_s=perf_counter()-t0
        max_obj=max(abs(float(a.objective)-float(b.objective)) for a,b in zip(sr,cr))
        rows.append({'n':n,'rep':rep,'steps':STEPS,'stateful_s':state_s,'cold_s':cold_s,'ratio':state_s/cold_s,'speedup':cold_s/state_s,'reuse_count':sum(v is True for v in reuse),'expected_reuse':STEPS-1,'max_objective_diff':max_obj,'all_valid':all(r.validation and r.validation.valid for r in sr+cr)})
    summary={}
    for n in SIZES:
      rr=[r for r in rows if r['n']==n]
      summary[str(n)]={'median_ratio':float(median(r['ratio'] for r in rr)),'median_speedup':float(median(r['speedup'] for r in rr)),'reuse_counts':[r['reuse_count'] for r in rr],'max_objective_diff':max(r['max_objective_diff'] for r in rr)}
    payload={'benchmark':'M10 CasADi-OSQP persistent solver object + explicit primal x0 warm-start','sizes':list(SIZES),'steps':STEPS,'repeats':REPEATS,'summary':summary,'environment':{'python':sys.version,'numpy':np.__version__,'scipy':scipy.__version__,'casadi':casadi.__version__,'platform':platform.platform(),'cpu_count':os.cpu_count()},'wall_s':perf_counter()-start,'rows':rows,'claim_boundary':'reuse evidence means same CasADi OSQP Function plus explicit x0 primal warm start; no factorization-reuse claim'}
    OUT.write_text(json.dumps(payload,indent=2),encoding='utf-8');print(json.dumps({'summary':summary,'wall_s':payload['wall_s']},indent=2))
if __name__=='__main__':main()
