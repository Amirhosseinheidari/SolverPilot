from __future__ import annotations
import json, random, time
from pathlib import Path
import numpy as np
from solverpilot import Model, PersistentSession, conform_persistent_backend
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
from solverpilot.runtime.executor import execute

OUT=Path('benchmarks/results/p5-persistent-stress.json')
rng=random.Random(20260906)

def highs_campaign(n=160):
    m=Model('p5-highs-stress')
    x=m.variable(3, lower=0.0, upper=10.0)
    c=m.parameter(3, value=[1.0,2.0,3.0], name='c')
    rhs=m.parameter(value=2.0, name='rhs')
    a=m.parameter(value=1.0, name='a')
    m.add(a*x[0] + x[1] + x[2] >= rhs)
    m.minimize(c @ x)
    backend=BundledHighsCAPIBackend(threads=1, solver='simplex')
    rep=conform_persistent_backend(backend); assert rep.passed
    sess=PersistentSession(m, backend, capabilities=rep.manifest)
    counts={}; max_obj=0.0; mismatches=0
    t0=time.perf_counter()
    for k in range(n):
        kind=rng.choice(['obj','rhs','matrix'])
        if kind=='obj':
            vals=np.array([rng.uniform(0.5,3.0) for _ in range(3)])
            c.value=vals
        elif kind=='rhs':
            rhs.value=rng.uniform(0.5,6.0)
        else:
            # keep nonzero to preserve sparsity for native patch campaign
            a.value=rng.uniform(0.5,2.0)
        out=sess.solve()
        counts[out.trace.path.value]=counts.get(out.trace.path.value,0)+1
        fresh=BundledHighsCAPIBackend(threads=1, solver='simplex')
        cold=execute(out.compiled.execution_ir, fresh)
        fresh.close()
        if out.result.objective is None or cold.objective is None:
            mismatches += 1
        else:
            e=abs(out.result.objective-cold.objective); max_obj=max(max_obj,e)
            if e>1e-8: mismatches += 1
        assert out.trace.original_validation_valid is True
    elapsed=time.perf_counter()-t0
    sess.close()
    return {'steps':n,'mismatches':mismatches,'max_objective_error':max_obj,'paths':counts,'wall_s':elapsed}

def osqp_campaign(n=160):
    m=Model('p5-osqp-stress')
    x=m.variable(2, lower=-5.0, upper=5.0)
    q=m.parameter(2, value=[-2.0,-4.0], name='q')
    rhs=m.parameter(value=8.0, name='rhs')
    p=m.parameter(value=1.0, name='p')
    m.add(x.sum() <= rhs)
    m.minimize(p*x[0]*x[0] + x[1]*x[1] + q @ x)
    backend=BundledOSQPCAPIBackend()
    rep=conform_persistent_backend(backend); assert rep.passed
    sess=PersistentSession(m, backend, capabilities=rep.manifest)
    counts={}; max_obj=0.0; max_x=0.0; mismatches=0
    t0=time.perf_counter()
    for k in range(n):
        kind=rng.choice(['obj','rhs','quad'])
        if kind=='obj': q.value=[rng.uniform(-4.0,0.0),rng.uniform(-5.0,-1.0)]
        elif kind=='rhs': rhs.value=rng.uniform(4.0,10.0)
        else: p.value=rng.uniform(0.5,2.5)
        out=sess.solve(); counts[out.trace.path.value]=counts.get(out.trace.path.value,0)+1
        fresh=BundledOSQPCAPIBackend(); cold=execute(out.compiled.execution_ir,fresh); fresh.close()
        if out.result.objective is None or cold.objective is None:
            mismatches += 1
        else:
            eo=abs(out.result.objective-cold.objective); max_obj=max(max_obj,eo)
            ex=float(np.max(np.abs(out.result.x-cold.x))); max_x=max(max_x,ex)
            if eo>1e-6 or ex>1e-5: mismatches += 1
        assert out.trace.original_validation_valid is True
    elapsed=time.perf_counter()-t0
    sess.close()
    return {'steps':n,'mismatches':mismatches,'max_objective_error':max_obj,'max_x_error':max_x,'paths':counts,'wall_s':elapsed}

payload={'schema':'solverpilot.p5.persistent-stress.v1','seed':20260906,'highs':highs_campaign(),'osqp':osqp_campaign()}
OUT.write_text(json.dumps(payload,indent=2,sort_keys=True))
print(json.dumps(payload,indent=2,sort_keys=True))
