from __future__ import annotations
import json,time
from pathlib import Path
import numpy as np
from solverpilot.model import Model, sin, cos, exp, tanh
from solverpilot.nlp import NLPDerivativeEngine, compile_nlp_model
OUT=Path(__file__).parent/'results'/'p7-nlp-derivative-campaign.json'

def main():
    rng=np.random.default_rng(7062026); grad_max=0.; jac_max=0.; hess_max=0.; passed=0
    t0=time.perf_counter()
    for k in range(500):
        m=Model(); x=m.variable(4,lower=-2,upper=2); a=rng.normal(size=4); b=rng.normal(size=4)
        m.minimize(((a*x).sum())**2+0.2*sin(x[0]*x[1])+0.1*cos(x[2])+0.05*exp(0.2*x[3])+0.03*tanh(x[1]))
        m.add((b*x).sum()+0.2*sin(x[0])+0.1*x[2]*x[3] <= 3.0)
        ad=NLPDerivativeEngine(compile_nlp_model(m).execution_ir); z=rng.uniform(-0.7,0.7,size=4); h=1e-6
        g=ad.gradient(z); J=ad.jacobian(z); gfd=np.zeros(4); Jfd=np.zeros((1,4))
        for i in range(4):
            zp=z.copy(); zm=z.copy(); zp[i]+=h; zm[i]-=h
            gfd[i]=(ad.objective(zp)-ad.objective(zm))/(2*h); Jfd[:,i]=(ad.constraints(zp)-ad.constraints(zm))/(2*h)
        ge=float(np.max(np.abs(g-gfd))); je=float(np.max(np.abs(J-Jfd))); grad_max=max(grad_max,ge); jac_max=max(jac_max,je)
        if k < 200:
            hh=2e-5; H=ad.lagrangian_hessian(z,np.zeros(1)); Hfd=np.zeros_like(H)
            for i in range(4):
                zp=z.copy(); zm=z.copy(); zp[i]+=hh; zm[i]-=hh
                Hfd[:,i]=(ad.gradient(zp)-ad.gradient(zm))/(2*hh)
            he=float(np.max(np.abs(H-Hfd))); hess_max=max(hess_max,he)
            ok=ge<5e-6 and je<5e-6 and he<5e-5
        else: ok=ge<5e-6 and je<5e-6
        passed += int(ok)
    payload={'schema':'solverpilot.p7.derivative-campaign.v1','cases':500,'hessian_cases':200,'passed':passed,'gradient_max_abs_error':grad_max,'jacobian_max_abs_error':jac_max,'hessian_max_abs_error':hess_max,'wall_s':time.perf_counter()-t0,'method':'CasADi AD cross-checked against central finite differences'}
    OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)+'\n'); print(json.dumps(payload,indent=2))
if __name__=='__main__': main()
