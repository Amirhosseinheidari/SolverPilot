from __future__ import annotations
import json, time
from pathlib import Path
import numpy as np
from solverpilot.model import Model, sin, cos

OUT=Path(__file__).parent/'results'/'p7-nlp-public-formulation-smoke.json'

def solve_case(name, model, starts, public_bound, source):
    rows=[]
    for x0 in starts:
        t=time.perf_counter(); r=model.solve(x0=x0); dt=time.perf_counter()-t
        rows.append({'x0':list(map(float,x0)),'status':r.backend_status,'objective':r.objective_reported,'x':None if r.x is None else r.x.tolist(),'valid':r.validation.valid,'local_optimal_candidate':r.local_optimal_candidate,'globally_proven':r.globally_proven,'wall_s':dt,'abs_to_public_bound':None if r.objective_reported is None else abs(float(r.objective_reported)-public_bound)})
    return {'instance':name,'source':source,'public_primal_bound':public_bound,'rows':rows}

def main():
    cases=[]
    m=Model('rbrock'); x=m.variable(2,lower=[-10,-10],upper=[5,10]); m.minimize(100*(x[1]-x[0]**2)**2+(1-x[0])**2)
    cases.append(solve_case('rbrock',m,[[-1.2,1.0]],[0.0][0],'https://minlplib.org/rbrock.html'))
    m=Model('trigx'); x=m.variable(2); m.minimize(x[0]**2+x[1]**2); m.add(x[0]-sin(2*x[0]+3*x[1])-cos(3*x[0]-5*x[1])==0); m.add(x[1]-sin(x[0]-2*x[1])+cos(x[0]+3*x[1])==0)
    cases.append(solve_case('trigx',m,[[0,0],[1,1]],[0.09563139][0],'https://minlplib.org/trigx.html'))
    m=Model('mathopt4'); x=m.variable(2,lower=-10,upper=10); m.minimize((2*x[0]**2-x[1]**2)**2+(x[1]-6*x[0]**2)**2); m.add(x[0]-(100*sin(2*x[0]+3*x[1])+10*x[1])==0); m.add(x[0]+x[1]<=2)
    cases.append(solve_case('mathopt4',m,[[0,0],[1,1],[-1,-1]],[0.0][0],'https://minlplib.org/mathopt4.html'))
    payload={'schema':'solverpilot.p7.nlp-public-formulation-smoke.v1','classification':'official-formulation-reproduction_not_official_archive_execution','cases':cases,'all_valid':all(row['valid'] for c in cases for row in c['rows']),'no_global_claims':all(not row['globally_proven'] for c in cases for row in c['rows'])}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n')
    print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
