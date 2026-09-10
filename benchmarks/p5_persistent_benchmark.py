from __future__ import annotations
import json, time, statistics
from pathlib import Path
import numpy as np
from solverpilot import Model, PersistentSession, conform_persistent_backend
from solverpilot.backends import BundledHighsCAPIBackend, BundledOSQPCAPIBackend
from solverpilot.runtime.executor import execute

OUT=Path('benchmarks/results/p5-persistent-benchmark.json')
rng=np.random.default_rng(20260906)

def med(xs): return float(statistics.median(xs))

def cold_semantic(model, caps, backend_factory):
    comp=model.compile(use_cache=False, capabilities=caps)
    b=backend_factory(); r=execute(comp.execution_ir,b)
    if r.x is not None: comp.validate_original(model,r.x)
    close=getattr(b,'close',None)
    if callable(close): close()
    return r

def highs_localized(n=200,m=80,reps=16):
    A=rng.uniform(0.0,1.0,size=(m,n)); base=.25*A.sum(axis=1)
    md=Model('p5-highs-localized'); x=md.variable(n,lower=0,upper=1)
    rhs0=md.parameter(value=float(base[0]),name='rhs0'); c=md.parameter(n,value=rng.uniform(.5,2,n),name='c')
    md.add(A[0]@x>=rhs0)
    for i in range(1,m): md.add(A[i]@x>=float(base[i]))
    md.minimize(c@x)
    b=BundledHighsCAPIBackend(threads=1,solver='simplex'); rep=conform_persistent_backend(b); assert rep.passed
    sess=PersistentSession(md,b,capabilities=rep.manifest); sess.solve(); ps=[]; cs=[]; errs=[]
    for _ in range(reps):
        rhs0.value=float(base[0]*(.9+.2*rng.random()))
        t=time.perf_counter(); pr=sess.solve(); ps.append(time.perf_counter()-t)
        t=time.perf_counter(); cr=cold_semantic(md,rep.manifest,lambda:BundledHighsCAPIBackend(threads=1,solver='simplex')); cs.append(time.perf_counter()-t)
        errs.append(abs(pr.objective-cr.objective))
    out={'n':n,'m':m,'repetitions':reps,'persistent_median_s':med(ps),'cold_median_s':med(cs),'persistent_over_cold':med(ps)/med(cs),'max_objective_error':max(errs),'compiler_cache_status':sess.history[-1].compiler_cache_status}
    sess.close(); return out

def highs_global(n=60,m=30,reps=16):
    A=rng.uniform(0.0,1.0,size=(m,n)); base=.25*A.sum(axis=1)
    md=Model('p5-highs-global'); x=md.variable(n,lower=0,upper=1)
    rhs=md.parameter(m,value=base,name='rhs'); c=md.parameter(n,value=rng.uniform(.5,2,n),name='c')
    md.add(A@x>=rhs); md.minimize(c@x)
    b=BundledHighsCAPIBackend(threads=1,solver='simplex'); rep=conform_persistent_backend(b); assert rep.passed
    sess=PersistentSession(md,b,capabilities=rep.manifest); sess.solve(); ps=[]; cs=[]; errs=[]
    for _ in range(reps):
        rhs.value=base*(.9+.2*rng.random(m))
        t=time.perf_counter(); pr=sess.solve(); ps.append(time.perf_counter()-t)
        t=time.perf_counter(); cr=cold_semantic(md,rep.manifest,lambda:BundledHighsCAPIBackend(threads=1,solver='simplex')); cs.append(time.perf_counter()-t)
        errs.append(abs(pr.objective-cr.objective))
    out={'n':n,'m':m,'repetitions':reps,'persistent_median_s':med(ps),'cold_median_s':med(cs),'persistent_over_cold':med(ps)/med(cs),'max_objective_error':max(errs),'compiler_cache_status':sess.history[-1].compiler_cache_status}
    sess.close(); return out

def osqp_objective(n=150,reps=16):
    md=Model('p5-osqp'); x=md.variable(n,lower=-5,upper=5); q0=rng.uniform(-2,2,n); q=md.parameter(n,value=q0,name='q')
    md.minimize((x*x).sum()+q@x)
    b=BundledOSQPCAPIBackend(); rep=conform_persistent_backend(b); assert rep.passed
    sess=PersistentSession(md,b,capabilities=rep.manifest); sess.solve(); ps=[]; cs=[]; errs=[]
    for _ in range(reps):
        q.value=q0+rng.normal(0,.2,n)
        t=time.perf_counter(); pr=sess.solve(); ps.append(time.perf_counter()-t)
        t=time.perf_counter(); cr=cold_semantic(md,rep.manifest,BundledOSQPCAPIBackend); cs.append(time.perf_counter()-t)
        errs.append(abs(pr.objective-cr.objective))
    out={'n':n,'repetitions':reps,'persistent_median_s':med(ps),'cold_median_s':med(cs),'persistent_over_cold':med(ps)/med(cs),'max_objective_error':max(errs),'compiler_cache_status':sess.history[-1].compiler_cache_status}
    sess.close(); return out

payload={
 'schema':'solverpilot.p5.persistent-benchmark.v2','seed':20260906,
 'timing_evidence':'single-host smoke only; medians are not release performance claims',
 'highs_localized_rhs':highs_localized(),
 'highs_global_rhs':highs_global(),
 'osqp_objective':osqp_objective(),
}
OUT.write_text(json.dumps(payload,indent=2,sort_keys=True)); print(json.dumps(payload,indent=2,sort_keys=True))
