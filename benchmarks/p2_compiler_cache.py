from __future__ import annotations
import json, statistics, time
from pathlib import Path
import numpy as np
from solverpilot.model import Model


def timed(fn, reps=15):
    vals=[]
    for _ in range(reps):
        t=time.perf_counter(); fn(); vals.append(time.perf_counter()-t)
    return {"median_s":statistics.median(vals),"min_s":min(vals),"max_s":max(vals),"repetitions":len(vals)}


def localized_rhs(seed=2):
    rng=np.random.default_rng(seed); n=400; mrows=120
    model=Model(); x=model.variable(n,lower=-5,upper=5)
    A=rng.normal(size=(mrows,n))
    ps=[model.parameter(value=float(rng.normal())) for _ in range(mrows)]
    for i,p in enumerate(ps): model.add(A[i]@x <= p)
    model.minimize(rng.normal(size=n)@x); model.compile()
    k=0
    def cached():
        nonlocal k; k=(k+1)%mrows; ps[k].value=float(k)+0.123; model.compile()
    k2=0
    def full():
        nonlocal k2; k2=(k2+1)%mrows; ps[k2].value=float(k2)+0.456; model.compile(use_cache=False)
    return timed(cached), timed(full)


def objective_only(seed=3):
    rng=np.random.default_rng(seed); n=500; mrows=40
    model=Model(); x=model.variable(n,lower=-5,upper=5)
    A=rng.normal(size=(mrows,n)); b=rng.normal(size=mrows)
    q=model.parameter(n,value=rng.normal(size=n)); model.add(A@x<=b); model.minimize(q@x); model.compile()
    def cached(): q.value=rng.normal(size=n); model.compile()
    def full(): q.value=rng.normal(size=n); model.compile(use_cache=False)
    return timed(cached), timed(full)


def global_dense(seed=4):
    rng=np.random.default_rng(seed); n=300; mrows=50
    model=Model(); x=model.variable(n,lower=-5,upper=5)
    A=model.parameter((mrows,n),value=rng.normal(size=(mrows,n))); b=model.parameter(mrows,value=rng.normal(size=mrows))
    q=rng.normal(size=n); model.add(A@x<=b); model.minimize(q@x); model.compile()
    def cached(): A.value=rng.normal(size=(mrows,n)); b.value=rng.normal(size=mrows); model.compile()
    def full(): A.value=rng.normal(size=(mrows,n)); b.value=rng.normal(size=mrows); model.compile(use_cache=False)
    return timed(cached,10), timed(full,10)


def snapshot(seed=5):
    rng=np.random.default_rng(seed); n=500; mrows=30
    model=Model(); x=model.variable(n,lower=-5,upper=5); p=model.parameter(value=1.)
    A=rng.normal(size=(mrows,n)); b=rng.normal(size=mrows); model.add(A@x<=b); model.minimize(rng.normal(size=n)@x+p)
    model.compile(); p.value=2.; model.compile(); p.value=1.; model.compile()
    def hit(): model.compile()
    return timed(hit,50)


def main():
    lr_c,lr_f=localized_rhs(); ob_c,ob_f=objective_only(); gd_c,gd_f=global_dense(); sh=snapshot()
    out={
      "schema":"solverpilot.p2.compiler-cache-benchmark.v1",
      "localized_rhs":{"cached":lr_c,"full":lr_f,"ratio_cached_to_full":lr_c["median_s"]/lr_f["median_s"]},
      "objective_only":{"cached":ob_c,"full":ob_f,"ratio_cached_to_full":ob_c["median_s"]/ob_f["median_s"]},
      "global_dense":{"cached":gd_c,"full":gd_f,"ratio_cached_to_full":gd_c["median_s"]/gd_f["median_s"]},
      "snapshot_hit":{"cached":sh},
      "claim_policy":"smoke evidence only; no production performance claim",
    }
    path=Path('benchmarks/results/p2-compiler-cache.json'); path.write_text(json.dumps(out,indent=2,sort_keys=True)+"\n")
    print(json.dumps(out,indent=2,sort_keys=True))
if __name__=='__main__': main()
