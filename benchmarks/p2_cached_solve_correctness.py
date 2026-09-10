from __future__ import annotations
import json
from pathlib import Path
import numpy as np
from solverpilot.model import Model
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
from solverpilot.validate import CandidateSolution, validate_solution


def main():
    rng=np.random.default_rng(62026)
    lp_ok=0
    m=Model(); x=m.variable(5,lower=0,upper=4)
    rhs=m.parameter(value=1.0); q=m.parameter(5,value=np.ones(5))
    m.add(np.ones(5)@x>=rhs); m.minimize(q@x); m.compile()
    highs=BundledHighsCAPIBackend()
    for _ in range(100):
        rhs.value=float(rng.uniform(0.1,8.0)); q.value=rng.uniform(0.2,3.0,size=5)
        cached=m.compile(); full=m.compile(use_cache=False)
        rc=cached.solve(backend=highs); rf=full.solve(backend=highs)
        vc=validate_solution(cached.execution_ir, CandidateSolution(rc.x, rc.objective))
        vf=validate_solution(full.execution_ir, CandidateSolution(rf.x, rf.objective))
        if not (vc.valid and vf.valid): raise AssertionError('LP validation failed')
        if abs(rc.objective-rf.objective)>1e-9: raise AssertionError('LP objective mismatch')
        lp_ok+=1

    qp_ok=0; max_x=0.0; max_obj=0.0
    m=Model(); x=m.variable(4,lower=-10,upper=10)
    d=m.parameter(4,value=np.ones(4),sign='nonnegative'); q=m.parameter(4,value=np.zeros(4))
    expr=q@x
    for i in range(4): expr=expr+0.5*d[i]*x[i]*x[i]
    m.minimize(expr); m.compile(); osqp=BundledOSQPCAPIBackend()
    for _ in range(120):
        dv=rng.uniform(0.5,4,size=4); qv=rng.normal(0,1,size=4)
        d.value=dv; q.value=qv
        cached=m.compile(); r=cached.solve(backend=osqp)
        expected=np.clip(-qv/dv,-10,10)
        expected_obj=float(0.5*np.dot(dv,expected*expected)+np.dot(qv,expected))
        xerr=float(np.max(np.abs(r.x-expected))); oerr=abs(float(r.objective)-expected_obj)
        max_x=max(max_x,xerr); max_obj=max(max_obj,oerr)
        if xerr>2e-5 or oerr>2e-6: raise AssertionError((xerr,oerr))
        qp_ok+=1
    out={'schema':'solverpilot.p2.cached-solve-correctness.v1','lp_cached_vs_full_solve':{'passed':lp_ok,'total':100},'qp_cached_vs_analytic':{'passed':qp_ok,'total':120,'max_x_abs_error':max_x,'max_objective_abs_error':max_obj}}
    path=Path('verification/2026-09-06/p2-cached-solve-correctness.json'); path.write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
