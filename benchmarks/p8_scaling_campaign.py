import json,time
from pathlib import Path
import numpy as np, solverpilot as om
from solverpilot.minlp import solve_binary_enumeration, solve_outer_approximation
rng=np.random.default_rng(80808); rows=[]; ok=0
for case in range(10):
    k=int(rng.integers(4,9)); m=om.Model(f'scale{case}'); x=m.variable(lower=-2,upper=2); z=m.binary(k)
    obj=(x-float(rng.uniform(-.5,.5)))**2
    targets=rng.uniform(.15,.85,size=k)
    for j in range(k): obj=obj+(z[j]-float(targets[j]))**2
    m.minimize(obj); p=m.compile().execution_ir
    t=time.perf_counter(); e=solve_binary_enumeration(p); et=time.perf_counter()-t
    t=time.perf_counter(); o=solve_outer_approximation(p,max_iter=300); ot=time.perf_counter()-t
    match=bool(e.globally_proven and o.globally_proven and abs(e.objective-o.objective)<=2e-5)
    ok+=match; rows.append({'case':case,'k':k,'enum':e.objective,'oa':o.objective,'match':match,'enum_s':et,'oa_s':ot,'oa_over_enum':ot/et,'iterations':len(o.iterations)})
out={'schema':'solverpilot.p8.scaling-campaign.v1','seed':80808,'cases':10,'passed':ok,'rows':rows}
Path('benchmarks/results/p8-scaling-campaign.json').write_text(json.dumps(out,indent=2))
print(json.dumps({'passed':ok,'cases':10,'median_k':float(np.median([r['k'] for r in rows])),'median_enum_s':float(np.median([r['enum_s'] for r in rows])),'median_oa_s':float(np.median([r['oa_s'] for r in rows])),'median_oa_over_enum':float(np.median([r['oa_over_enum'] for r in rows])),'max_iter':max(r['iterations'] for r in rows)},indent=2))
