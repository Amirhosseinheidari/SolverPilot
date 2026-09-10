from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from solverpilot.model import Model
OUT=Path(__file__).parent/'results'/'p7-nlp-analytic-campaign.json'

def main():
    rng=np.random.default_rng(77); rows=[]; t0=time.perf_counter()
    max_x=0.; max_obj=0.; passed=0
    for k in range(120):
        c=rng.uniform(-1,1,size=2); m=Model(); x=m.variable(2,lower=-3,upper=3)
        m.minimize((x[0]-c[0])**4+(x[1]-c[1])**4+0.2*((x[0]-c[0])**2+(x[1]-c[1])**2))
        r=m.solve(x0=[0,0]); xe=float(np.max(np.abs(r.x-c))); oe=abs(float(r.objective_reported)); max_x=max(max_x,xe); max_obj=max(max_obj,oe); ok=r.validation.valid and r.local_optimal_candidate and not r.globally_proven and xe<5e-4; passed+=int(ok)
    # constrained projection to unit circle in positive quadrant, analytic [1/sqrt(2),1/sqrt(2)]
    target=np.array([1/np.sqrt(2)]*2)
    constrained=0; cmax=0.
    for k in range(80):
        m=Model(); x=m.variable(2,lower=0,upper=2); m.minimize((x[0]-1)**2+(x[1]-1)**2); m.add(x[0]**2+x[1]**2 <= 1)
        r=m.solve(x0=[0.2+0.005*k,0.3]); err=float(np.max(np.abs(r.x-target))); cmax=max(cmax,err); constrained += int(r.validation.valid and r.local_optimal_candidate and not r.globally_proven and err<8e-5)
    payload={'schema':'solverpilot.p7.analytic-campaign.v1','unconstrained_cases':120,'unconstrained_passed':passed,'max_x_error':max_x,'max_objective_error':max_obj,'constrained_cases':80,'constrained_passed':constrained,'constrained_max_x_error':cmax,'global_claims':0,'wall_s':time.perf_counter()-t0}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
