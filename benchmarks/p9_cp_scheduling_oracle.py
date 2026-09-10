from __future__ import annotations
import itertools,json,time
from pathlib import Path
import numpy as np, solverpilot as om
rng=np.random.default_rng(9072026); rows=[]; passed=0; t0=time.perf_counter()
for case in range(120):
    nt=int(rng.integers(2,5)); horizon=int(rng.integers(3,7)); sizes=[int(x) for x in rng.integers(1,3,size=nt)]; demands=[int(x) for x in rng.integers(1,4,size=nt)]; cap=int(rng.integers(max(demands),max(demands)+4))
    m=om.CPModel(); starts=[m.int_var(0,horizon-sizes[i]) for i in range(nt)]; ints=[m.interval_var(starts[i],sizes[i]) for i in range(nt)]
    mode='no_overlap' if rng.random()<.5 else 'cumulative'
    if mode=='no_overlap': m.add_no_overlap(ints)
    else: m.add_cumulative(ints,demands,cap)
    m.minimize(sum(starts)); r=m.solve(om.ReferenceCPBackend(max_states=1_000_000))
    best=None
    for ss in itertools.product(*(s.domain for s in starts)):
        spans=[(ss[i],ss[i]+sizes[i]) for i in range(nt)]
        if mode=='no_overlap':
            ok=all(e1<=s2 or e2<=s1 for i,(s1,e1) in enumerate(spans) for s2,e2 in spans[i+1:])
        else:
            points=sorted(set(s for s,e in spans)); ok=all(sum(demands[i] for i,(s,e) in enumerate(spans) if s<=t<e)<=cap for t in points)
        if ok:
            val=sum(ss); best=val if best is None or val<best else best
    match=(best is None and r.status=='infeasible') or (best is not None and r.optimality_proven and r.validation.valid and r.objective==best)
    passed+=int(match); rows.append({'case':case,'mode':mode,'oracle':best,'solverpilot':r.objective,'match':match,'states':r.raw_statistics.get('states_total')})
out={'schema':'solverpilot.p9.cp-scheduling-oracle.v1','seed':9072026,'cases':120,'passed':passed,'wall_s':time.perf_counter()-t0,'rows':rows}
Path('benchmarks/results/p9-cp-scheduling-oracle.json').write_text(json.dumps(out,indent=2,sort_keys=True)+'\n'); print(json.dumps({'cases':120,'passed':passed,'wall_s':out['wall_s']},indent=2))
