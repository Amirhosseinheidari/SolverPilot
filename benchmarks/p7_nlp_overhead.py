from __future__ import annotations
import json,time,statistics
from pathlib import Path
import numpy as np
from solverpilot.model import Model, sin
from solverpilot.nlp import NLPDerivativeEngine, compile_nlp_model
OUT=Path(__file__).parent/'results'/'p7-nlp-overhead.json'

def build(n):
    m=Model(); x=m.variable(n,lower=-2,upper=2); obj=0
    for i in range(n-1): obj=obj+100*(x[i+1]-x[i]**2)**2+(1-x[i])**2
    obj=obj+0.01*sin(x.sum()); m.minimize(obj)
    for i in range(min(10,n)): m.add(x[i]**2 <= 4)
    return m

def main():
    rows=[]
    for n in [10,50,100]:
        builds=[]; evals=[]
        for _ in range(5):
            m=build(n); c=compile_nlp_model(m); t=time.perf_counter(); ad=NLPDerivativeEngine(c.execution_ir); builds.append(time.perf_counter()-t); z=np.zeros(n); t=time.perf_counter(); ad.evaluate(z); ad.lagrangian_hessian(z,np.zeros(c.execution_ir.n_constraints)); evals.append(time.perf_counter()-t)
        rows.append({'n':n,'ad_build_median_s':statistics.median(builds),'eval_grad_jac_hess_median_s':statistics.median(evals)})
    payload={'schema':'solverpilot.p7.overhead.v1','classification':'overhead_characterization_not_solver_speedup','rows':rows}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
