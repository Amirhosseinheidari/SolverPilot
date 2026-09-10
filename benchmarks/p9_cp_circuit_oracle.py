from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np, solverpilot as om
rng=np.random.default_rng(9907); rows=[]; passed=0; t0=time.perf_counter()
def graph_ok(arcs,bits):
    selected=[(t,h) for (t,h),b in zip(arcs,bits) if b]; nodes=set(x for e in arcs for x in e)
    if any(sum(t==n for t,h in selected)!=1 or sum(h==n for t,h in selected)!=1 for n in nodes): return False
    ns=[e for e in selected if e[0]!=e[1]]
    if not ns: return True
    active=set(x for e in ns for x in e); start=next(iter(active)); cur=start; seen={start}
    for _ in range(len(active)):
        nxt=[h for t,h in ns if t==cur]
        if len(nxt)!=1: return False
        cur=nxt[0]
        if cur==start: break
        if cur in seen: return False
        seen.add(cur)
    return cur==start and seen==active
for case in range(80):
    n=3; arcs=[(i,i) for i in range(n)] + [(0,1),(1,2),(2,0),(0,2),(2,1),(1,0)]
    costs=[int(x) for x in rng.integers(-3,7,size=len(arcs))]
    m=om.CPModel(); bs=[m.bool_var() for _ in arcs]; m.add_circuit([(t,h,b) for (t,h),b in zip(arcs,bs)]); m.minimize(sum(costs[i]*bs[i] for i in range(len(bs))))
    r=m.solve(om.ReferenceCPBackend(max_states=10000)); best=None
    for bits in itertools.product((0,1),repeat=len(arcs)):
        if not graph_ok(arcs,bits): continue
        v=sum(c*b for c,b in zip(costs,bits)); best=v if best is None or v<best else best
    match=r.optimality_proven and r.validation.valid and r.objective==best; passed+=int(match); rows.append({'case':case,'match':match,'oracle':best,'solverpilot':r.objective})
out={'schema':'solverpilot.p9.cp-circuit-oracle.v1','seed':9907,'cases':80,'passed':passed,'wall_s':time.perf_counter()-t0,'rows':rows}
Path('benchmarks/results/p9-cp-circuit-oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'cases':80,'passed':passed,'wall_s':out['wall_s']},indent=2))
